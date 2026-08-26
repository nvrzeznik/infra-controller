/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use hw_platform::{ChassisIdentity, ServiceIdentity};
use nv_redfish::core::Bmc;
use nv_redfish::{Resource, ServiceRoot};

use crate::HealthError;
use crate::endpoint::{BmcAddr, BmcEndpoint, BmcPlatform, EndpointMetadata};

struct SystemIdentity {
    id: String,
    uuid: Option<uuid::Uuid>,
    bios_version: Option<String>,
}

fn select_primary_system(systems: &[SystemIdentity]) -> Option<&SystemIdentity> {
    systems
        .iter()
        .find(|system| {
            system
                .bios_version
                .as_deref()
                .is_some_and(|version| !version.trim().is_empty())
        })
        .or_else(|| systems.first())
}

/// Resolves the endpoint identity discovery owns: the primary ComputerSystem
/// UUID, and the hardware platform.
///
/// Both are answered from one `ServiceRoot` fetch. Each is stored in write-once
/// shared state, so a result reached after collectors started still propagates
/// to them, and an endpoint whose identity is already known makes no request at
/// all.
///
/// Failure is the caller's to log and swallow: identity is enrichment, and an
/// endpoint whose BMC would not answer must still be collected from. An
/// unresolved cell is retried on the next discovery pass.
pub(super) async fn ensure_endpoint_identity(endpoint: &BmcEndpoint) -> Result<(), HealthError> {
    // The switch host side speaks NVUE/gNMI and exposes no Redfish service;
    // probing it spends a connection attempt per pass to learn nothing.
    if !endpoint.supports_redfish() {
        return Ok(());
    }

    let machine = match endpoint.metadata.as_ref() {
        Some(EndpointMetadata::Machine(machine)) => Some(machine),
        _ => None,
    };

    let system_uuid_pending = machine.is_some_and(|machine| !machine.system_uuid.initialized());
    if !system_uuid_pending && endpoint.platform.initialized() {
        return Ok(());
    }

    let root = ServiceRoot::new(endpoint.bmc().clone()).await?;
    let primary_system = primary_system_identity(&root, &endpoint.addr).await;

    // An unreadable ComputerSystem collection is permanent on some endpoints:
    // both power shelves answer `/redfish/v1/Systems` with 404, and holding
    // their platform hostage to it would leave them unidentified forever.
    // Classification proceeds without a system id where the id provably cannot
    // change the answer, and is deferred to the next pass where it could --
    // guessing there would cache a confidently wrong platform.
    let vendor = root.vendor().map(|value| value.into_inner());
    let system_id_available = primary_system.is_ok() || !hw_platform::needs_system_id(vendor);

    if system_id_available {
        endpoint
            .platform
            .get_or_try_init(|| async {
                let primary = primary_system.as_ref().ok().and_then(Option::as_ref);
                resolve_platform(&root, primary, &endpoint.addr).await
            })
            .await?;
    }

    // The UUID gets no such fallback. An unreadable collection is not evidence
    // that a machine has no UUID, so the error propagates and the cell is left
    // uninitialized for the next pass to retry.
    let primary_system = primary_system?;

    if let Some(machine) = machine {
        machine
            .system_uuid
            .get_or_try_init(|| async {
                let Some(primary) = primary_system.as_ref() else {
                    return Ok(None);
                };
                if primary.uuid.is_none() {
                    tracing::warn!(
                        bmc_address = ?endpoint.addr,
                        system_id = %primary.id,
                        "Primary ComputerSystem does not expose a UUID"
                    );
                }
                Ok::<Option<uuid::Uuid>, HealthError>(primary.uuid)
            })
            .await?;
    }

    Ok(())
}

/// The ComputerSystem that identifies this endpoint, or `None` when the BMC
/// exposes no usable system collection.
///
/// A system with a non-empty BIOS version is preferred because BMCs may expose
/// auxiliary systems alongside the host. When no system has BIOS metadata, the
/// first collection member is used.
async fn primary_system_identity<B: Bmc + 'static>(
    root: &ServiceRoot<B>,
    addr: &BmcAddr,
) -> Result<Option<SystemIdentity>, HealthError> {
    let Some(systems) = root.systems().await? else {
        // Both power shelves are like this, so it is not on its own a fault.
        tracing::debug!(
            bmc_address = ?addr,
            "BMC does not expose a ComputerSystem collection"
        );
        return Ok(None);
    };

    let systems = systems.members().await?;
    let identities: Vec<SystemIdentity> = systems
        .iter()
        .map(|system| {
            let raw = system.raw();
            SystemIdentity {
                id: raw.base.id.clone(),
                uuid: raw.uuid.flatten(),
                bios_version: raw.bios_version.clone().flatten(),
            }
        })
        .collect();

    if identities.is_empty() {
        tracing::warn!(
            bmc_address = ?addr,
            "BMC exposes an empty ComputerSystem collection"
        );
    }

    Ok(
        select_primary_system(&identities).map(|primary| SystemIdentity {
            id: primary.id.clone(),
            uuid: primary.uuid,
            bios_version: primary.bios_version.clone(),
        }),
    )
}

/// Classifies the hardware platform, keeping the raw vendor and product strings
/// whether or not classification succeeded.
///
/// The chassis collection is fetched because it is load-bearing, not
/// supplementary: a DGX GB300 tray and a GB200 tray have byte-identical service
/// roots, and only the NVIDIA GB300 GPU chassis tells them apart. A fetch
/// *error* is propagated so the endpoint is retried rather than pinned to a
/// classification made without that evidence; a BMC that exposes no chassis
/// collection at all is a fact, and classification proceeds without it.
async fn resolve_platform<B: Bmc + 'static>(
    root: &ServiceRoot<B>,
    primary_system: Option<&SystemIdentity>,
    addr: &BmcAddr,
) -> Result<BmcPlatform, HealthError> {
    let chassis = match root.chassis().await? {
        Some(collection) => collection.members().await?,
        None => {
            tracing::debug!(
                bmc_address = ?addr,
                "BMC does not expose a Chassis collection"
            );
            Vec::new()
        }
    };
    let chassis: Vec<ChassisIdentity<'_>> = chassis
        .iter()
        .map(|chassis| {
            let hardware_id = chassis.hardware_id();
            ChassisIdentity {
                id: chassis.id().into_inner(),
                manufacturer: hardware_id.manufacturer.map(|value| value.into_inner()),
                model: hardware_id.model.map(|value| value.into_inner()),
            }
        })
        .collect();

    let vendor = root.vendor().map(|value| value.into_inner());
    let product = root.product().map(|value| value.into_inner());
    let hw_type = hw_platform::classify(
        ServiceIdentity {
            vendor,
            product,
            oem_id: root.oem_id().map(|value| value.into_inner()),
            // The BIOS-bearing system, not merely the first: on a BMC that
            // exposes an auxiliary baseboard beside the host, the host is the
            // one whose id names the platform.
            system_id: primary_system.map(|system| system.id.as_str()),
        },
        &chassis,
    );

    if hw_type.is_none() {
        tracing::debug!(
            bmc_address = ?addr,
            bmc_vendor = ?vendor,
            bmc_product = ?product,
            "BMC reports no recognized hardware platform"
        );
    }

    Ok(BmcPlatform {
        hw_type,
        vendor: vendor.and_then(non_empty),
        product: product.and_then(non_empty),
    })
}

/// Trims, and treats an all-whitespace value as unreported. BMCs pad these
/// fields, and a blank string would otherwise publish as a real answer.
fn non_empty(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIRST_UUID: uuid::Uuid = uuid::uuid!("11111111-1111-1111-1111-111111111111");
    const BIOS_UUID: uuid::Uuid = uuid::uuid!("22222222-2222-2222-2222-222222222222");

    #[test]
    fn primary_system_prefers_first_system_with_bios() {
        let systems = [
            SystemIdentity {
                id: "auxiliary".to_string(),
                uuid: Some(FIRST_UUID),
                bios_version: None,
            },
            SystemIdentity {
                id: "host".to_string(),
                uuid: Some(BIOS_UUID),
                bios_version: Some("1.2.3".to_string()),
            },
        ];

        let primary = select_primary_system(&systems).expect("primary system");

        assert_eq!(primary.id, "host");
        assert_eq!(primary.uuid, Some(BIOS_UUID));
    }

    #[test]
    fn primary_system_falls_back_to_first_member() {
        let systems = [
            SystemIdentity {
                id: "first".to_string(),
                uuid: Some(FIRST_UUID),
                bios_version: None,
            },
            SystemIdentity {
                id: "second".to_string(),
                uuid: Some(BIOS_UUID),
                bios_version: Some("  ".to_string()),
            },
        ];

        let primary = select_primary_system(&systems).expect("primary system");

        assert_eq!(primary.id, "first");
        assert_eq!(primary.uuid, Some(FIRST_UUID));
    }

    #[test]
    fn primary_system_prefers_host_bios_over_auxiliary_uuid() {
        let systems = [
            SystemIdentity {
                id: "HGX_Baseboard".to_string(),
                uuid: Some(FIRST_UUID),
                bios_version: None,
            },
            SystemIdentity {
                id: "host".to_string(),
                uuid: None,
                bios_version: Some("1.2.3".to_string()),
            },
        ];

        let primary = select_primary_system(&systems).expect("primary system");

        assert_eq!(primary.id, "host");
        assert_eq!(primary.uuid, None);
    }

    #[test]
    fn primary_system_is_absent_for_empty_collection() {
        assert!(select_primary_system(&[]).is_none());
    }
}

/// End-to-end platform resolution against `bmc-mock`, which serves the real
/// service-root, system, and chassis payloads for each platform.
///
/// The unit table in `hw-platform` pins the classification rules; this pins the
/// projection onto them -- that the fields are read from the resources the BMC
/// actually serves, and that the chassis collection is fetched at all.
#[cfg(test)]
mod bmc_mock_integration_tests {
    use std::str::FromStr;

    use bmc_mock::test_support::{
        TestBmcHandle, dell_poweredge_r750_bmc, delta_powershelf_bmc, dgx_gb300_bmc,
        generic_ami_bmc, generic_supermicro_bmc, hpe_proliant_dl380a_gen11_bmc, lenovo_gb300_bmc,
        liteon_powershelf_bmc, nvidia_dgx_vr_host_bmc, nvidia_switch_nd5200_ld_bmc,
        supermicro_gb300_bmc, wiwynn_gb200_bmc,
    };
    use hw_platform::HwType;
    use mac_address::MacAddress;

    use super::*;

    fn addr() -> BmcAddr {
        BmcAddr {
            ip: "10.0.0.1".parse().expect("valid ip"),
            port: Some(443),
            mac: MacAddress::from_str("00:11:22:33:44:55").expect("valid mac"),
        }
    }

    async fn platform_of(handle: TestBmcHandle) -> BmcPlatform {
        let root = handle.service_root;
        let addr = addr();
        // Mirrors production: an unreadable system collection is tolerated, and
        // the power shelves rely on that -- they answer `/Systems` with 404.
        let primary = primary_system_identity(&root, &addr).await;

        resolve_platform(&root, primary.as_ref().ok().and_then(Option::as_ref), &addr)
            .await
            .expect("platform resolves")
    }

    async fn hw_type_of(handle: TestBmcHandle) -> Option<HwType> {
        platform_of(handle).await.hw_type
    }

    #[tokio::test]
    async fn resolves_host_platforms() {
        assert_eq!(
            hw_type_of(wiwynn_gb200_bmc().await).await,
            Some(HwType::Gb200)
        );
        assert_eq!(
            hw_type_of(nvidia_dgx_vr_host_bmc().await).await,
            Some(HwType::VeraRubin)
        );
        assert_eq!(
            hw_type_of(dell_poweredge_r750_bmc().await).await,
            Some(HwType::Dell)
        );
        assert_eq!(
            hw_type_of(hpe_proliant_dl380a_gen11_bmc().await).await,
            Some(HwType::Hpe)
        );
        assert_eq!(hw_type_of(generic_ami_bmc().await).await, Some(HwType::Ami));
        assert_eq!(
            hw_type_of(generic_supermicro_bmc().await).await,
            Some(HwType::Supermicro)
        );
    }

    // Every GB300 variant shares its service root with something else, so each
    // one is proof the chassis collection was fetched and read.
    #[tokio::test]
    async fn resolves_gb300_variants_that_need_the_chassis() {
        assert_eq!(
            hw_type_of(dgx_gb300_bmc().await).await,
            Some(HwType::DgxGb300),
            "DGX GB300 shares GB200's service root exactly"
        );
        assert_eq!(
            hw_type_of(lenovo_gb300_bmc().await).await,
            Some(HwType::LenovoGb300),
            "Lenovo GB300 reports a generic AMI service root"
        );
        assert_eq!(
            hw_type_of(supermicro_gb300_bmc().await).await,
            Some(HwType::SupermicroGb300),
        );
    }

    // Neither shelf exposes a Systems collection, so resolution has to survive
    // its absence and fall through to the chassis.
    #[tokio::test]
    async fn resolves_endpoints_without_a_system_collection() {
        assert_eq!(
            hw_type_of(liteon_powershelf_bmc().await).await,
            Some(HwType::LiteonPowerShelf)
        );
        assert_eq!(
            hw_type_of(delta_powershelf_bmc().await).await,
            Some(HwType::DeltaPowerShelf)
        );
    }

    #[tokio::test]
    async fn resolves_switch_bmc_platforms() {
        assert_eq!(
            hw_type_of(nvidia_switch_nd5200_ld_bmc().await).await,
            Some(HwType::NvSwitch)
        );
    }

    // The raw pair is what keeps an unclassified platform identifiable, so it
    // has to be populated even where classification succeeds.
    #[tokio::test]
    async fn keeps_the_raw_vendor_and_product_beside_the_classification() {
        let platform = platform_of(wiwynn_gb200_bmc().await).await;

        assert_eq!(platform.hw_type, Some(HwType::Gb200));
        assert_eq!(platform.vendor.as_deref(), Some("WIWYNN"));
        assert_eq!(platform.product.as_deref(), Some("GB200 NVL"));
        assert!(!platform.is_empty());
    }
}
