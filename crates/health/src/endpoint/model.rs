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

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::future::Future;
use std::net::IpAddr;
use std::sync::Arc;

use carbide_uuid::machine::MachineId;
use carbide_uuid::nvlink::NvLinkDomainId;
use carbide_uuid::power_shelf::PowerShelfId;
use carbide_uuid::rack::RackId;
use carbide_uuid::switch::SwitchId;
use hw_platform::HwType;
use mac_address::MacAddress;
use tokio::sync::OnceCell;
use url::Url;

use crate::HealthError;
use crate::bmc::{BmcClient, BoxFuture};

/// Shared, write-once UUID reported by a machine's primary ComputerSystem.
///
/// Collectors clone endpoint metadata when they start, so this state must remain
/// shared for a UUID resolved after collector startup to reach emitted events.
/// Clones of the same state compare equal. Distinct states compare equal only
/// after both cells are initialized with the same optional UUID.
#[derive(Clone, Debug, Default)]
pub struct SharedSystemUuid(Arc<OnceCell<Option<uuid::Uuid>>>);

impl PartialEq for SharedSystemUuid {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || matches!((self.0.get(), other.0.get()), (Some(left), Some(right)) if left == right)
    }
}

impl SharedSystemUuid {
    pub fn get(&self) -> Option<uuid::Uuid> {
        self.0.get().copied().flatten()
    }

    /// True once resolution has run, whether or not it found a UUID.
    ///
    /// [`get`](Self::get) cannot answer this: it returns `None` both for "not
    /// yet asked" and for "asked, and this BMC reports no UUID".
    pub(crate) fn initialized(&self) -> bool {
        self.0.initialized()
    }

    pub(crate) async fn get_or_try_init<E, F, Fut>(&self, f: F) -> Result<Option<uuid::Uuid>, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Option<uuid::Uuid>, E>>,
    {
        self.0.get_or_try_init(f).await.copied()
    }
}

impl From<Option<uuid::Uuid>> for SharedSystemUuid {
    fn from(system_uuid: Option<uuid::Uuid>) -> Self {
        match system_uuid {
            Some(system_uuid) => Self(Arc::new(OnceCell::new_with(Some(Some(system_uuid))))),
            None => Self::default(),
        }
    }
}

/// What a BMC reports about the hardware it manages.
///
/// The raw `vendor` and `product` strings are kept beside the classified
/// `hw_type` rather than discarded once classification succeeds. A platform
/// that reaches the fleet before its classifier arm does still emits events,
/// and those events stay identifiable: `hw_type` is absent, but the raw pair
/// names the hardware, so triage works and a downstream rule can be written
/// against it without waiting for a hw-health release. With only `hw_type`, an
/// unrecognized platform would be indistinguishable from an unreachable BMC.
///
/// A fully empty value is a real outcome -- the BMC answered and identified
/// nothing -- and is distinct from the unresolved state that [`SharedPlatform`]
/// represents with an uninitialized cell.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BmcPlatform {
    /// The classified platform, absent when no classifier arm matched.
    pub hw_type: Option<HwType>,
    /// `ServiceRoot.Vendor`, trimmed; absent when empty or unreported.
    pub vendor: Option<String>,
    /// `ServiceRoot.Product`, trimmed; absent when empty or unreported.
    pub product: Option<String>,
}

impl BmcPlatform {
    /// True when the BMC identified itself in no way at all.
    ///
    /// Publishing reads the three fields individually -- a platform that
    /// reported only a vendor still publishes that vendor -- so this is not a
    /// gate on emission. It names the "answered, and said nothing" outcome,
    /// which is what distinguishes a resolved-but-empty cell from an
    /// unresolved one.
    pub fn is_empty(&self) -> bool {
        self.hw_type.is_none() && self.vendor.is_none() && self.product.is_none()
    }
}

/// Shared, write-once hardware platform for one BMC endpoint.
///
/// The same shape and the same reasoning as [`SharedSystemUuid`]: collectors
/// clone endpoint state when they start and a streaming collector keeps that
/// clone for the life of its stream, so a platform resolved after collector
/// startup only reaches emitted events through shared state.
///
/// This lives on [`BmcEndpoint`] rather than in [`MachineData`] because it is
/// not machine-specific. Switch BMCs emit Redfish events too, and the taxonomy
/// has switch and power-shelf variants; hanging it off machine metadata would
/// silently drop every non-machine endpoint.
#[derive(Clone, Debug, Default)]
pub struct SharedPlatform(Arc<OnceCell<BmcPlatform>>);

impl PartialEq for SharedPlatform {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || matches!((self.0.get(), other.0.get()), (Some(left), Some(right)) if left == right)
    }
}

impl SharedPlatform {
    pub fn get(&self) -> Option<&BmcPlatform> {
        self.0.get()
    }

    /// True once resolution has run, whether or not it identified anything.
    pub(crate) fn initialized(&self) -> bool {
        self.0.initialized()
    }

    pub(crate) async fn get_or_try_init<E, F, Fut>(&self, f: F) -> Result<&BmcPlatform, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<BmcPlatform, E>>,
    {
        self.0.get_or_try_init(f).await
    }
}

#[derive(Clone)]
pub struct BmcEndpoint {
    pub addr: BmcAddr,
    pub metadata: Option<EndpointMetadata>,
    pub rack_id: Option<RackId>,
    pub labels: BTreeMap<String, String>,
    pub bmc: Arc<BmcClient>,

    /// Hardware platform reported by this BMC, resolved on demand by discovery.
    ///
    /// Shared write-once state so a platform resolved after collectors start
    /// still reaches their emitted events, and so both a present and an absent
    /// result are cached and the BMC is queried only once.
    pub platform: SharedPlatform,
}

impl BmcEndpoint {
    pub fn key(&self) -> String {
        self.addr.mac.to_string()
    }

    pub fn hash_key(&self) -> Cow<'static, str> {
        Cow::Owned(
            self.rack_id
                .as_ref()
                .map(|id| id.to_string())
                .unwrap_or_else(|| self.key()),
        )
    }

    pub fn log_identity(&self) -> Cow<'_, str> {
        match &self.metadata {
            Some(EndpointMetadata::Machine(MachineData {
                machine_id: Some(id),
                ..
            })) => Cow::Owned(id.to_string()),
            Some(EndpointMetadata::PowerShelf(power_shelf)) => Cow::Borrowed(&power_shelf.serial),
            Some(EndpointMetadata::Switch(switch)) => Cow::Borrowed(&switch.serial),
            _ => Cow::Owned(self.addr.mac.to_string()),
        }
    }

    /// Returns whether this endpoint speaks Redfish at all.
    ///
    /// Every endpoint kind does except the switch *host* side, which is reached
    /// over NVUE/gNMI and exposes no Redfish service. Probing it would spend a
    /// connection attempt and a warning per discovery pass to learn nothing.
    pub(crate) fn supports_redfish(&self) -> bool {
        !matches!(
            self.metadata.as_ref(),
            Some(EndpointMetadata::Switch(switch)) if switch.endpoint_role == SwitchEndpointRole::Host
        )
    }

    /// Returns whether this endpoint supports periodic Redfish log collection.
    pub(crate) fn supports_periodic_logs(&self) -> bool {
        match self.metadata.as_ref() {
            Some(EndpointMetadata::Machine(_)) => true,
            Some(EndpointMetadata::Switch(switch)) => {
                switch.endpoint_role == SwitchEndpointRole::Bmc
            }
            Some(EndpointMetadata::PowerShelf(_)) => {
                // Power shelves may expose compatible LogServices, but behavior depends on
                // hardware and firmware. Keep collection disabled until future implementation
                // and validation establish support.
                false
            }
            None => false,
        }
    }

    pub fn bmc(&self) -> &Arc<BmcClient> {
        &self.bmc
    }

    pub fn switch_data(&self) -> Option<&SwitchData> {
        self.metadata.as_ref().and_then(EndpointMetadata::as_switch)
    }

    /// Returns the connect host direct switch collectors should place in URIs.
    ///
    /// Switch collectors connect to the discovered endpoint IP address. DNS
    /// names used for TLS verification are handled separately.
    pub fn switch_connect_host_for_uri(&self) -> Cow<'_, str> {
        match self.addr.ip {
            IpAddr::V4(ip) => Cow::Owned(ip.to_string()),
            IpAddr::V6(ip) => Cow::Owned(format!("[{ip}]")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum EndpointMetadata {
    Machine(MachineData),
    PowerShelf(PowerShelfData),
    Switch(SwitchData),
}

impl EndpointMetadata {
    pub fn as_switch(&self) -> Option<&SwitchData> {
        match self {
            EndpointMetadata::Switch(switch) => Some(switch),
            _ => None,
        }
    }

    pub fn serial_number(&self) -> Option<&str> {
        match self {
            EndpointMetadata::Machine(machine) => machine.machine_serial.as_deref(),
            EndpointMetadata::PowerShelf(power_shelf) => Some(power_shelf.serial.as_str()),
            EndpointMetadata::Switch(switch) => Some(switch.serial.as_str()),
        }
    }

    /// Returns the PHR component category represented by this endpoint metadata.
    pub const fn component_type(&self) -> &'static str {
        match self {
            Self::Machine(_) => "compute_node",
            Self::PowerShelf(_) => "power_shelf",
            Self::Switch(_) => "nvlink_switch",
        }
    }
}

/// Metadata that describes a machine endpoint for health telemetry.
#[derive(Clone, Debug, PartialEq)]
pub struct MachineData {
    /// Stable NICo machine identifier. None when running without NICo.
    pub machine_id: Option<MachineId>,

    /// Hardware chassis serial discovered from machine DMI data, when known.
    pub machine_serial: Option<String>,

    /// UUID reported by the primary Redfish ComputerSystem resource.
    ///
    /// Endpoint discovery resolves this on demand. The write-once shared state
    /// propagates enrichment to collectors that already started and records
    /// both present and absent UUID results so successful BMC queries happen
    /// only once.
    pub system_uuid: SharedSystemUuid,

    /// Physical rack slot where the machine is installed, when known.
    pub slot_number: Option<i32>,

    /// Compute tray index where the machine is installed, when known.
    pub tray_index: Option<i32>,

    /// NVLink domain UUID for the machine, when it participates in an NVLink domain.
    pub nvlink_domain_uuid: Option<NvLinkDomainId>,

    /// Machine-level GPU driver version.
    ///
    /// This is populated only when API discovery reports exactly one unique
    /// non-empty GPU driver version for the machine. It stays absent when the
    /// version is unknown or the discovered GPUs report conflicting versions.
    pub driver_version: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PowerShelfData {
    pub id: Option<PowerShelfId>,
    pub serial: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwitchEndpointRole {
    Bmc,
    Host,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwitchData {
    pub id: Option<SwitchId>,
    pub serial: String,
    pub slot_number: Option<i32>,
    pub tray_index: Option<i32>,

    /// NVLink domain UUID associated with the switch, when known.
    ///
    /// Discovery restarts collectors when this value changes so subsequent
    /// telemetry uses current metadata.
    pub nvlink_domain_uuid: Option<NvLinkDomainId>,

    pub endpoint_role: SwitchEndpointRole,
    pub is_primary: bool,
    pub nmxc_enabled: bool,
    pub nmxt_enabled: bool,
}

#[derive(Clone)]
pub enum BmcCredentials {
    UsernamePassword {
        username: String,
        password: Option<String>,
    },
    SessionToken {
        token: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct BmcAddr {
    pub ip: IpAddr,
    pub port: Option<u16>,
    pub mac: MacAddress,
}

impl BmcAddr {
    /// Builds the BMC base URL. IPv6 literals are bracketed so the URL
    /// authority parses — a bare `IpAddr` Display leaves IPv6 unbracketed,
    /// which `Url::parse` would otherwise reject.
    pub fn to_url(&self) -> Result<Url, url::ParseError> {
        let scheme = if self.port.is_some_and(|v| v == 80) {
            "http"
        } else {
            "https"
        };
        // Bracket IPv6 hosts; IPv4 renders unchanged.
        let host = match self.ip {
            IpAddr::V4(v4) => v4.to_string(),
            IpAddr::V6(v6) => format!("[{v6}]"),
        };
        let mut url = Url::parse(&format!("{scheme}://{host}"))?;
        let _ = url.set_port(self.port);
        Ok(url)
    }
}

impl From<BmcCredentials> for nv_redfish::bmc_http::BmcCredentials {
    fn from(value: BmcCredentials) -> Self {
        match value {
            BmcCredentials::UsernamePassword { username, password } => {
                nv_redfish::bmc_http::BmcCredentials::username_password(username, password)
            }
            BmcCredentials::SessionToken { token } => {
                nv_redfish::bmc_http::BmcCredentials::token(token)
            }
        }
    }
}

pub trait EndpointSource: Send + Sync {
    fn fetch_bmc_hosts<'a>(&'a self) -> BoxFuture<'a, Result<Vec<Arc<BmcEndpoint>>, HealthError>>;
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::net::IpAddr;
    use std::str::FromStr;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use carbide_test_support::{Check, check_values};
    use mac_address::MacAddress;

    use super::{
        BmcAddr, BmcCredentials, BmcPlatform, EndpointMetadata, MachineData, PowerShelfData,
        SharedPlatform, SharedSystemUuid, SwitchData, SwitchEndpointRole,
    };
    use crate::endpoint::test_support::{endpoint_with_creds, mac, test_endpoint};

    fn addr(ip: &str, port: Option<u16>) -> BmcAddr {
        BmcAddr {
            ip: IpAddr::from_str(ip).unwrap(),
            port,
            mac: MacAddress::from_str("00:11:22:33:44:55").unwrap(),
        }
    }

    // A v6 BMC IP must render as a bracketed URL authority, else Url::parse rejects it.
    #[test]
    fn to_url_brackets_ipv6() {
        let url = addr("2001:db8::1", Some(443)).to_url().unwrap();
        assert_eq!(url.scheme(), "https");
        assert_eq!(url.host_str(), Some("[2001:db8::1]"));
    }

    // v4 hosts are byte-identical to the old behaviour.
    #[test]
    fn to_url_v4_unchanged() {
        let url = addr("10.0.0.5", Some(443)).to_url().unwrap();
        assert_eq!(url.scheme(), "https");
        assert_eq!(url.host_str(), Some("10.0.0.5"));
    }

    // Port 80 selects the http scheme (v6 still bracketed).
    #[test]
    fn to_url_port_80_is_http() {
        let url = addr("2001:db8::1", Some(80)).to_url().unwrap();
        assert_eq!(url.scheme(), "http");
        assert_eq!(url.host_str(), Some("[2001:db8::1]"));
    }

    #[test]
    fn switch_connect_host_for_uri_brackets_ipv6() {
        let endpoint = endpoint_with_creds(
            addr("2001:db8::1", Some(443)),
            BmcCredentials::UsernamePassword {
                username: "admin".to_string(),
                password: Some("pass".to_string()),
            },
            None,
            None,
        );

        assert_eq!(endpoint.switch_connect_host_for_uri(), "[2001:db8::1]");
    }

    #[test]
    fn periodic_log_collection_endpoint_eligibility() {
        let switch_bmc = SwitchData {
            id: None,
            serial: "switch".to_string(),
            slot_number: Some(1),
            tray_index: Some(2),
            nvlink_domain_uuid: None,
            endpoint_role: SwitchEndpointRole::Bmc,
            is_primary: true,
            nmxc_enabled: false,
            nmxt_enabled: false,
        };

        let switch_host = SwitchData {
            endpoint_role: SwitchEndpointRole::Host,
            ..switch_bmc.clone()
        };

        check_values(
            [
                Check {
                    scenario: "machine BMC is eligible",
                    input: Some(EndpointMetadata::Machine(MachineData {
                        machine_id: None,
                        machine_serial: None,
                        system_uuid: SharedSystemUuid::default(),
                        slot_number: None,
                        tray_index: None,
                        nvlink_domain_uuid: None,
                        driver_version: None,
                    })),
                    expect: true,
                },
                Check {
                    scenario: "switch BMC is eligible",
                    input: Some(EndpointMetadata::Switch(switch_bmc)),
                    expect: true,
                },
                Check {
                    scenario: "switch host is not eligible",
                    input: Some(EndpointMetadata::Switch(switch_host)),
                    expect: false,
                },
                Check {
                    scenario: "power shelf is not eligible",
                    input: Some(EndpointMetadata::PowerShelf(PowerShelfData {
                        id: None,
                        serial: "power-shelf".to_string(),
                    })),
                    expect: false,
                },
                Check {
                    scenario: "endpoint without metadata is not eligible",
                    input: None,
                    expect: false,
                },
            ],
            |metadata| {
                let mut endpoint = test_endpoint(mac("00:11:22:33:44:55"));
                endpoint.metadata = metadata;

                endpoint.supports_periodic_logs()
            },
        );
    }

    // Identity resolution probes Redfish, so it has to skip the one endpoint
    // kind that has none. Every other kind emits Redfish events and needs a
    // platform -- the taxonomy has switch and power-shelf variants.
    #[test]
    fn redfish_probing_covers_every_endpoint_kind_except_the_switch_host() {
        let switch_bmc = SwitchData {
            id: None,
            serial: "switch".to_string(),
            slot_number: Some(1),
            tray_index: Some(2),
            nvlink_domain_uuid: None,
            endpoint_role: SwitchEndpointRole::Bmc,
            is_primary: true,
            nmxc_enabled: false,
            nmxt_enabled: false,
        };

        let switch_host = SwitchData {
            endpoint_role: SwitchEndpointRole::Host,
            ..switch_bmc.clone()
        };

        check_values(
            [
                Check {
                    scenario: "machine BMC speaks Redfish",
                    input: Some(EndpointMetadata::Machine(MachineData {
                        machine_id: None,
                        machine_serial: None,
                        system_uuid: SharedSystemUuid::default(),
                        slot_number: None,
                        tray_index: None,
                        nvlink_domain_uuid: None,
                        driver_version: None,
                    })),
                    expect: true,
                },
                Check {
                    scenario: "switch BMC speaks Redfish",
                    input: Some(EndpointMetadata::Switch(switch_bmc)),
                    expect: true,
                },
                // NVUE/gNMI only; there is no Redfish service to ask.
                Check {
                    scenario: "switch host does not",
                    input: Some(EndpointMetadata::Switch(switch_host)),
                    expect: false,
                },
                // Unlike periodic log collection, which power shelves opt out
                // of, identity resolution covers them -- the taxonomy names
                // both shelf vendors and only the chassis identifies them.
                Check {
                    scenario: "power shelf speaks Redfish",
                    input: Some(EndpointMetadata::PowerShelf(PowerShelfData {
                        id: None,
                        serial: "power-shelf".to_string(),
                    })),
                    expect: true,
                },
                Check {
                    scenario: "an endpoint without metadata is assumed to",
                    input: None,
                    expect: true,
                },
            ],
            |metadata| {
                let mut endpoint = test_endpoint(mac("00:11:22:33:44:55"));
                endpoint.metadata = metadata;

                endpoint.supports_redfish()
            },
        );
    }

    #[tokio::test]
    async fn shared_platform_caches_an_empty_result_across_clones() {
        let state = SharedPlatform::default();
        let clone = state.clone();
        let query_count = AtomicUsize::new(0);

        for state in [&state, &clone] {
            state
                .get_or_try_init(|| async {
                    query_count.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, Infallible>(BmcPlatform::default())
                })
                .await
                .expect("infallible platform initialization");
        }

        // A BMC that identified nothing is still an answer; asking again every
        // discovery pass would spend requests to re-learn it.
        assert_eq!(query_count.load(Ordering::SeqCst), 1);
        assert!(state.initialized());
        assert!(state.get().is_some_and(BmcPlatform::is_empty));
        assert!(clone.get().is_some_and(BmcPlatform::is_empty));
    }

    #[tokio::test]
    async fn shared_system_uuid_caches_absent_result_across_clones() {
        let state = SharedSystemUuid::default();
        let clone = state.clone();
        let query_count = AtomicUsize::new(0);

        for state in [&state, &clone] {
            state
                .get_or_try_init(|| async {
                    query_count.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, Infallible>(None)
                })
                .await
                .expect("infallible UUID initialization");
        }

        assert_eq!(query_count.load(Ordering::SeqCst), 1);
        assert_eq!(state.get(), None);
        assert_eq!(clone.get(), None);
    }

    #[tokio::test]
    async fn shared_system_uuid_equality_distinguishes_unresolved_from_absent() {
        let unresolved = SharedSystemUuid::default();
        let other_unresolved = SharedSystemUuid::default();

        assert_eq!(unresolved, unresolved.clone());
        assert_ne!(unresolved, other_unresolved);

        let initialized_absent = SharedSystemUuid::default();
        initialized_absent
            .get_or_try_init(|| async { Ok::<_, Infallible>(None) })
            .await
            .expect("infallible UUID initialization");

        assert_ne!(initialized_absent, unresolved);
    }
}
