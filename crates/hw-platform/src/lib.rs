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

//! Hardware platform taxonomy, and the rules that resolve one from Redfish.
//!
//! [`HwType`] names a hardware platform; [`classify`] resolves one from the
//! Redfish fields that identify it. Both live here rather than in a consumer so
//! that every caller answers "which platform is this?" the same way. Two tables
//! that answer the same question drift silently -- nothing fails when a new
//! platform is added to one and not the other, and the divergence surfaces
//! later as mislabelled telemetry.
//!
//! [`classify`] takes plain string fields rather than Redfish resource types so
//! callers with very different amounts of the BMC already fetched can share it:
//! `bmc-explorer` projects its exploration types onto them, `carbide-health`
//! projects a `ServiceRoot`, a `ComputerSystem`, and a `Chassis` collection.

use std::fmt;

/// A hardware platform.
///
/// This is coarser than a model number and finer than a vendor: it names a
/// class of machine whose Redfish surface, BIOS attributes, and event
/// vocabulary behave alike. Several variants share a [`bmc_vendor`] --
/// `Gb200` and `DgxGb300` are both NVIDIA -- which is why platform and vendor
/// are separate axes.
///
/// [`bmc_vendor`]: HwType::bmc_vendor
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HwType {
    Ami,
    Bluefield,
    Dell,
    Gb200,
    DgxGb300,
    Hpe,
    Lenovo,
    LenovoAmi,
    LenovoGb300,
    SupermicroGb300,
    Supermicro,
    Viking,
    LiteonPowerShelf,
    DeltaPowerShelf,
    NvSwitch,
    VeraRubin,
}

impl HwType {
    /// The platform's stable wire name.
    ///
    /// Emitted as the `hw.platform` OTLP attribute, so downstream rules key on
    /// these strings and they are an API. Spelled out rather than derived from
    /// the variant name so a Rust-side rename cannot silently change what the
    /// fleet reports; `wire_names_are_stable` pins every one.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ami => "ami",
            Self::Bluefield => "bluefield",
            Self::Dell => "dell",
            Self::Gb200 => "gb200",
            Self::DgxGb300 => "dgx_gb300",
            Self::Hpe => "hpe",
            Self::Lenovo => "lenovo",
            Self::LenovoAmi => "lenovo_ami",
            Self::LenovoGb300 => "lenovo_gb300",
            Self::SupermicroGb300 => "supermicro_gb300",
            Self::Supermicro => "supermicro",
            Self::Viking => "viking",
            Self::LiteonPowerShelf => "liteon_power_shelf",
            Self::DeltaPowerShelf => "delta_power_shelf",
            Self::NvSwitch => "nv_switch",
            Self::VeraRubin => "vera_rubin",
        }
    }

    pub const fn bmc_vendor(&self) -> Option<bmc_vendor::BMCVendor> {
        match self {
            Self::Ami => None,
            Self::Bluefield => Some(bmc_vendor::BMCVendor::Nvidia),
            Self::Dell => Some(bmc_vendor::BMCVendor::Dell),
            Self::Gb200 => Some(bmc_vendor::BMCVendor::Nvidia),
            // DGX GB300 uses the NVIDIA "GB BMC" (same BMC family as GB200).
            Self::DgxGb300 => Some(bmc_vendor::BMCVendor::Nvidia),
            Self::Hpe => Some(bmc_vendor::BMCVendor::Hpe),
            Self::Lenovo => Some(bmc_vendor::BMCVendor::Lenovo),
            Self::LenovoAmi => Some(bmc_vendor::BMCVendor::LenovoAMI),
            Self::LenovoGb300 => Some(bmc_vendor::BMCVendor::LenovoAMI),
            // SMC GB300 runs a Supermicro (OpenBMC) host BMC.
            Self::SupermicroGb300 => Some(bmc_vendor::BMCVendor::Supermicro),
            Self::LiteonPowerShelf => Some(bmc_vendor::BMCVendor::Liteon),
            Self::DeltaPowerShelf => Some(bmc_vendor::BMCVendor::Delta),
            Self::NvSwitch => Some(bmc_vendor::BMCVendor::Nvidia),
            Self::Supermicro => Some(bmc_vendor::BMCVendor::Supermicro),
            Self::Viking => Some(bmc_vendor::BMCVendor::Nvidia),
            Self::VeraRubin => Some(bmc_vendor::BMCVendor::Nvidia),
        }
    }

    pub const fn infinite_boot_enabled_attr(&self) -> Option<BiosAttr<'static>> {
        match self {
            Self::Ami => Some(BiosAttr::new_str("EndlessBoot", "Enabled")),
            Self::Bluefield => None,
            Self::Dell => Some(BiosAttr::new_str("BootSeqRetry", "Enabled")),
            Self::Gb200 => Some(BiosAttr::new_str("EmbeddedUefiShell", "Disabled")),
            // The DGX GB300 BIOS exposes EmbeddedUefiShell, but the value that means
            // infinite-boot-enabled is not yet characterized on hardware (GB200's polarity
            // is not assumed to carry over). Left None until confirmed on a tray.
            // TODO(dgx-gb300): set the infinite-boot attribute from the DGX GB300 BIOS.
            Self::DgxGb300 => None,
            Self::Hpe => None,
            Self::Lenovo => Some(BiosAttr::new_str("BootModes_InfiniteBootRetry", "Enabled")),
            Self::LenovoAmi => Some(BiosAttr::new_str("EndlessBoot", "Enabled")),
            Self::LenovoGb300 => Some(BiosAttr::new_int("LEM0003", 50)),
            // TODO(smc): confirm the SMC GB300 infinite-boot BIOS attribute from the tray BIOS.
            Self::SupermicroGb300 => None,
            Self::LiteonPowerShelf => None,
            Self::DeltaPowerShelf => None,
            Self::NvSwitch => None,
            Self::Supermicro => None,
            Self::Viking => Some(BiosAttr::new_str("NvidiaInfiniteboot", "Enable")),
            // Same EmbeddedUefiShell polarity as GB200 / libredfish NvidiaGBx00.
            Self::VeraRubin => Some(BiosAttr::new_str("EmbeddedUefiShell", "Disabled")),
        }
    }
}

impl fmt::Display for HwType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy)]
pub struct BiosAttr<'a> {
    pub key: &'a str,
    pub value: BiosAttrValue<'a>,
}

impl BiosAttr<'_> {
    pub const fn new_bool(key: &'static str, value: bool) -> BiosAttr<'static> {
        BiosAttr {
            key,
            value: BiosAttrValue::Bool(value),
        }
    }
    pub const fn new_str(key: &'static str, value: &'static str) -> BiosAttr<'static> {
        BiosAttr {
            key,
            value: BiosAttrValue::Str(value),
        }
    }
    pub const fn new_any_str(
        key: &'static str,
        value: &'static [&'static str],
    ) -> BiosAttr<'static> {
        BiosAttr {
            key,
            value: BiosAttrValue::AnyStr(value),
        }
    }
    pub const fn new_int(key: &'static str, value: i64) -> BiosAttr<'static> {
        BiosAttr {
            key,
            value: BiosAttrValue::Int(value),
        }
    }
}

#[derive(Clone, Copy)]
pub enum BiosAttrValue<'a> {
    Str(&'a str),
    AnyStr(&'a [&'a str]),
    Bool(bool),
    Int(i64),
}

impl fmt::Display for BiosAttrValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BiosAttrValue::Str(v) => v.fmt(f),
            BiosAttrValue::Bool(v) => v.fmt(f),
            BiosAttrValue::Int(v) => v.fmt(f),
            BiosAttrValue::AnyStr(v) => {
                write!(f, "any(")?;
                for (index, value) in v.iter().enumerate() {
                    if index > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{value}")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// The service-level identity a BMC reports, as [`classify`] reads it.
///
/// `vendor`, `product`, and `oem_id` come from the Redfish `ServiceRoot`;
/// `system_id` is the `Id` of the primary `ComputerSystem`, absent when the BMC
/// exposes no system collection (both power shelves do this).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ServiceIdentity<'a> {
    pub vendor: Option<&'a str>,
    pub product: Option<&'a str>,
    pub oem_id: Option<&'a str>,
    pub system_id: Option<&'a str>,
}

/// One `Chassis` member, as [`classify`] reads it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChassisIdentity<'a> {
    pub id: &'a str,
    pub manufacturer: Option<&'a str>,
    pub model: Option<&'a str>,
}

/// Resolves the hardware platform, or `None` when nothing identifies it.
///
/// `chassis` may be empty. Passing it empty is not free of consequence: the
/// GB300 platforms and both power shelves are *only* identifiable from chassis
/// data, and a GB300 tray with no chassis passed resolves as [`HwType::Gb200`]
/// rather than as `None`, because it shares GB200's service-root signature
/// exactly. Callers that cannot fetch the chassis collection should treat the
/// result as unreliable rather than as absent.
pub fn classify(service: ServiceIdentity<'_>, chassis: &[ChassisIdentity<'_>]) -> Option<HwType> {
    // GB300 is an NVIDIA HGX platform identity, recognized by the NVIDIA "NVIDIA GB300"
    // GPU chassis (`is_gb300()`) independent of the host BMC vendor. Resolve it before the
    // host-vendor match below so platform classification is not gated on the host ODM; the
    // ODM only selects the ODM-specific variant.
    if is_gb300(chassis) {
        // Lenovo GB300: AMI host BMC + Lenovo host chassis.
        if is_lenovo(chassis) {
            return Some(HwType::LenovoGb300);
        }
        // DGX GB300: NVIDIA "GB BMC" host (same BMC family as GB200). Resolved here, ahead of
        // the GB200 arm below, since it shares GB200's ServiceRoot signature -- the GB300 GPU
        // chassis (`is_gb300()`) is what distinguishes it from a real GB200.
        if service.vendor == Some("NVIDIA") && service.product == Some("GB BMC") {
            return Some(HwType::DgxGb300);
        }
        // SMC GB300: Supermicro OpenBMC host.
        if service.vendor == Some("Supermicro") {
            return Some(HwType::SupermicroGb300);
        }
    }

    service
        .vendor
        .or_else(|| (service.oem_id == Some("Supermicro")).then_some("Supermicro"))
        .and_then(|vendor_id| match vendor_id {
            "AMI" if service.system_id == Some("DGX") => Some(HwType::Viking),
            "AMI" => Some(HwType::Ami),
            "Dell" => Some(HwType::Dell),
            "Lenovo" if service.oem_id == Some("Ami") => Some(HwType::LenovoAmi),
            "Lenovo" if service.oem_id != Some("Ami") => Some(HwType::Lenovo),
            "Supermicro" => Some(HwType::Supermicro),
            "HPE" => Some(HwType::Hpe),
            "Nvidia" if is_bluefield_system_id(service.system_id) => Some(HwType::Bluefield),
            "NVIDIA" if service.product == Some("VR NVL72") => Some(HwType::VeraRubin),
            "WIWYNN" | "NVIDIA"
                if matches!(service.product, Some("GB200 NVL") | Some("GB BMC")) =>
            {
                Some(HwType::Gb200)
            }
            "NVIDIA" if service.product == Some("P3809") => Some(HwType::NvSwitch),
            _ => None,
        })
        .or_else(|| is_liteon_powershelf(chassis).then_some(HwType::LiteonPowerShelf))
        .or_else(|| is_delta_powershelf(chassis).then_some(HwType::DeltaPowerShelf))
}

/// True when classification for this vendor turns on the primary system's id.
///
/// Only two vendors are ambiguous without it: `AMI` splits Viking from generic
/// AMI, and `Nvidia` splits BlueField from unclassified. Everything else reaches
/// the same answer whether or not a system id was available.
///
/// This exists for callers that could not read the system collection at all --
/// both power shelves permanently 404 that path. It tells "the id is irrelevant
/// here, classify anyway" apart from "classifying without it would produce a
/// confidently wrong answer, so do not". `classification_ignores_system_id_for_other_vendors`
/// pins it against [`classify`] so the two cannot disagree.
pub fn needs_system_id(vendor: Option<&str>) -> bool {
    matches!(vendor, Some("AMI") | Some("Nvidia"))
}

/// True when any chassis member is the NVIDIA GB300 GPU chassis.
///
/// This is what separates a GB300 tray from a GB200 one; their service roots are
/// identical on the DGX variant.
fn is_gb300(chassis: &[ChassisIdentity<'_>]) -> bool {
    chassis
        .iter()
        .any(|c| c.manufacturer == Some("NVIDIA") && c.model == Some("NVIDIA GB300"))
}

fn is_lenovo(chassis: &[ChassisIdentity<'_>]) -> bool {
    chassis.iter().any(|c| c.manufacturer == Some("Lenovo"))
}

fn is_liteon_powershelf(chassis: &[ChassisIdentity<'_>]) -> bool {
    chassis.iter().any(|c| {
        c.id == "powershelf"
            || (c.id == "chassis"
                && c.manufacturer
                    .is_some_and(|mfg| mfg.to_lowercase().contains("lite-on")))
    })
}

/// Detects a Delta power shelf. Delta BMCs expose neither a `Vendor` in the
/// service root nor a `/redfish/v1/Systems` collection, so classification
/// relies on a Delta manufacturer on the power-shelf chassis (id "chassis"
/// or "powershelf"). The manufacturer gate is what distinguishes Delta from
/// the Lite-On power shelf, which shares the generic "powershelf" chassis id.
pub fn is_delta_powershelf(chassis: &[ChassisIdentity<'_>]) -> bool {
    chassis
        .iter()
        .any(|c| is_delta_powershelf_chassis(c.id, c.manufacturer))
}

/// Delta power-shelf identity gate: a power-shelf chassis (id `chassis` or
/// `powershelf`) whose manufacturer identifies as Delta. This is what
/// distinguishes a Delta shelf from the Lite-On shelf, which shares the generic
/// `powershelf` chassis id but reports a different manufacturer. Split out so
/// the gate can be exercised in unit tests without a live BMC.
fn is_delta_powershelf_chassis(chassis_id: &str, manufacturer: Option<&str>) -> bool {
    (chassis_id == "chassis" || chassis_id == "powershelf")
        && manufacturer.is_some_and(|mfg| mfg.to_lowercase().contains("delta"))
}

fn is_bluefield_system_id(system_id: Option<&str>) -> bool {
    matches!(system_id, Some("Bluefield") | Some("BlueField_0"))
}

#[cfg(test)]
mod tests {
    use bmc_vendor::BMCVendor;
    use carbide_test_support::{Check, check_values, value_scenarios};

    use super::*;

    /// Every variant, so a new platform cannot be added without deciding its
    /// wire name.
    const ALL: [HwType; 16] = [
        HwType::Ami,
        HwType::Bluefield,
        HwType::Dell,
        HwType::Gb200,
        HwType::DgxGb300,
        HwType::Hpe,
        HwType::Lenovo,
        HwType::LenovoAmi,
        HwType::LenovoGb300,
        HwType::SupermicroGb300,
        HwType::Supermicro,
        HwType::Viking,
        HwType::LiteonPowerShelf,
        HwType::DeltaPowerShelf,
        HwType::NvSwitch,
        HwType::VeraRubin,
    ];

    const GB300_CHASSIS: ChassisIdentity<'static> = ChassisIdentity {
        id: "HGX_Chassis_0",
        manufacturer: Some("NVIDIA"),
        model: Some("NVIDIA GB300"),
    };

    const LENOVO_CHASSIS: ChassisIdentity<'static> = ChassisIdentity {
        id: "Baseboard",
        manufacturer: Some("Lenovo"),
        model: None,
    };

    fn service<'a>(vendor: Option<&'a str>, product: Option<&'a str>) -> ServiceIdentity<'a> {
        ServiceIdentity {
            vendor,
            product,
            ..Default::default()
        }
    }

    // The wire names are published as `hw.platform`; a rename here is a fleet-visible
    // API change, so it has to be a deliberate edit to this table rather than a
    // side effect of renaming a Rust variant.
    #[test]
    fn wire_names_are_stable() {
        value_scenarios!(run = |hardware_type: HwType| hardware_type.as_str();
            "hardware types render stable wire names" {
                HwType::Ami => "ami",
                HwType::Bluefield => "bluefield",
                HwType::Dell => "dell",
                HwType::Gb200 => "gb200",
                HwType::DgxGb300 => "dgx_gb300",
                HwType::Hpe => "hpe",
                HwType::Lenovo => "lenovo",
                HwType::LenovoAmi => "lenovo_ami",
                HwType::LenovoGb300 => "lenovo_gb300",
                HwType::SupermicroGb300 => "supermicro_gb300",
                HwType::Supermicro => "supermicro",
                HwType::Viking => "viking",
                HwType::LiteonPowerShelf => "liteon_power_shelf",
                HwType::DeltaPowerShelf => "delta_power_shelf",
                HwType::NvSwitch => "nv_switch",
                HwType::VeraRubin => "vera_rubin",
            }
        );
    }

    // Two platforms sharing a wire name would silently merge downstream.
    #[test]
    fn wire_names_are_unique() {
        let mut names: Vec<&str> = ALL.iter().map(|hw| hw.as_str()).collect();
        names.sort_unstable();
        let unique = names.len();
        names.dedup();

        assert_eq!(names.len(), unique, "duplicate hw.platform wire name");
    }

    #[test]
    fn hw_type_bmc_vendor_maps_each_variant() {
        value_scenarios!(run = |hardware_type: HwType| hardware_type.bmc_vendor();
            "generic AMI has no canonical vendor" {
                HwType::Ami => None,
            }

            "hardware types map to canonical vendors" {
                HwType::Bluefield => Some(BMCVendor::Nvidia),
                HwType::Dell => Some(BMCVendor::Dell),
                HwType::Gb200 => Some(BMCVendor::Nvidia),
                HwType::DgxGb300 => Some(BMCVendor::Nvidia),
                HwType::Hpe => Some(BMCVendor::Hpe),
                HwType::Lenovo => Some(BMCVendor::Lenovo),
                HwType::LenovoAmi => Some(BMCVendor::LenovoAMI),
                HwType::LenovoGb300 => Some(BMCVendor::LenovoAMI),
                HwType::SupermicroGb300 => Some(BMCVendor::Supermicro),
                HwType::Supermicro => Some(BMCVendor::Supermicro),
                HwType::Viking => Some(BMCVendor::Nvidia),
                HwType::LiteonPowerShelf => Some(BMCVendor::Liteon),
                HwType::DeltaPowerShelf => Some(BMCVendor::Delta),
                HwType::NvSwitch => Some(BMCVendor::Nvidia),
                HwType::VeraRubin => Some(BMCVendor::Nvidia),
            }
        );
    }

    // The service-root signatures are bmc-mock's per-`HardwareType` ground truth
    // (`crates/bmc-mock/src/machine_info.rs`), which is what the integration
    // mocks actually serve.
    #[test]
    fn classifies_platforms_from_the_service_root_alone() {
        check_values(
            [
                Check {
                    scenario: "Dell iDRAC reports no product",
                    input: service(Some("Dell"), None),
                    expect: Some(HwType::Dell),
                },
                Check {
                    scenario: "HPE iLO",
                    input: service(Some("HPE"), Some("ProLiant DL380a Gen11")),
                    expect: Some(HwType::Hpe),
                },
                Check {
                    scenario: "Wiwynn ODM GB200 NVL tray reports its own vendor",
                    input: service(Some("WIWYNN"), Some("GB200 NVL")),
                    expect: Some(HwType::Gb200),
                },
                Check {
                    scenario: "NVIDIA GB BMC without GB300 chassis is a real GB200",
                    input: service(Some("NVIDIA"), Some("GB BMC")),
                    expect: Some(HwType::Gb200),
                },
                Check {
                    scenario: "Vera Rubin",
                    input: service(Some("NVIDIA"), Some("VR NVL72")),
                    expect: Some(HwType::VeraRubin),
                },
                Check {
                    scenario: "NVLink switch",
                    input: service(Some("NVIDIA"), Some("P3809")),
                    expect: Some(HwType::NvSwitch),
                },
                Check {
                    scenario: "generic AMI BMC",
                    input: service(Some("AMI"), Some("AMI Redfish Server")),
                    expect: Some(HwType::Ami),
                },
                Check {
                    scenario: "generic Supermicro",
                    input: service(Some("Supermicro"), Some("Super Server")),
                    expect: Some(HwType::Supermicro),
                },
                Check {
                    scenario: "unrecognised vendor",
                    input: service(Some("Acme"), Some("Anvil")),
                    expect: None,
                },
                Check {
                    scenario: "no vendor at all",
                    input: service(None, None),
                    expect: None,
                },
            ],
            |identity| classify(identity, &[]),
        );
    }

    // Viking and Bluefield share their vendor with platforms they must not be
    // confused with; the primary system's id is the discriminator.
    #[test]
    fn classifies_platforms_needing_the_primary_system_id() {
        check_values(
            [
                Check {
                    scenario: "DGX H100 (Viking) is an AMI BMC with a DGX system",
                    input: Some("DGX"),
                    expect: Some(HwType::Viking),
                },
                Check {
                    scenario: "any other AMI system is generic AMI",
                    input: Some("system"),
                    expect: Some(HwType::Ami),
                },
                Check {
                    scenario: "no system collection is generic AMI",
                    input: None,
                    expect: Some(HwType::Ami),
                },
            ],
            |system_id| {
                classify(
                    ServiceIdentity {
                        vendor: Some("AMI"),
                        product: Some("AMI Redfish Server"),
                        system_id,
                        ..Default::default()
                    },
                    &[],
                )
            },
        );

        check_values(
            [
                Check {
                    scenario: "BlueField-3 system id",
                    input: Some("Bluefield"),
                    expect: Some(HwType::Bluefield),
                },
                Check {
                    scenario: "BlueField_0 system id",
                    input: Some("BlueField_0"),
                    expect: Some(HwType::Bluefield),
                },
                Check {
                    scenario: "Nvidia vendor without a BlueField system is unclassified",
                    input: Some("system"),
                    expect: None,
                },
            ],
            |system_id| {
                classify(
                    ServiceIdentity {
                        vendor: Some("Nvidia"),
                        product: Some("BlueField-3 DPU"),
                        system_id,
                        ..Default::default()
                    },
                    &[],
                )
            },
        );
    }

    // The case Tier-2 resolution gets wrong. DGX GB300 and GB200 share a service
    // root byte for byte; only the GB300 GPU chassis tells them apart, and
    // reporting a GB300 tray as `gb200` is acted on rather than investigated.
    #[test]
    fn gb300_is_distinguished_from_gb200_only_by_the_chassis() {
        let dgx = service(Some("NVIDIA"), Some("GB BMC"));

        assert_eq!(classify(dgx, &[]), Some(HwType::Gb200));
        assert_eq!(classify(dgx, &[GB300_CHASSIS]), Some(HwType::DgxGb300));
    }

    #[test]
    fn classifies_gb300_odm_variants_from_the_chassis() {
        check_values(
            [
                Check {
                    scenario: "Lenovo GB300: AMI host BMC, Lenovo host chassis",
                    input: (
                        service(Some("AMI"), Some("AMI Redfish Server")),
                        &[GB300_CHASSIS, LENOVO_CHASSIS][..],
                    ),
                    expect: Some(HwType::LenovoGb300),
                },
                Check {
                    scenario: "DGX GB300: NVIDIA GB BMC host",
                    input: (
                        service(Some("NVIDIA"), Some("GB BMC")),
                        &[GB300_CHASSIS][..],
                    ),
                    expect: Some(HwType::DgxGb300),
                },
                Check {
                    scenario: "SMC GB300: Supermicro OpenBMC host",
                    input: (
                        service(Some("Supermicro"), Some("GB NVL")),
                        &[GB300_CHASSIS][..],
                    ),
                    expect: Some(HwType::SupermicroGb300),
                },
                // The GB300 arm only selects the ODM variant. An unknown host
                // vendor falls through to the vendor match rather than guessing.
                Check {
                    scenario: "GB300 chassis behind an unrecognised host vendor",
                    input: (service(Some("Acme"), None), &[GB300_CHASSIS][..]),
                    expect: None,
                },
                Check {
                    scenario: "a Supermicro without the GB300 chassis stays generic",
                    input: (service(Some("Supermicro"), Some("Super Server")), &[][..]),
                    expect: Some(HwType::Supermicro),
                },
            ],
            |(identity, chassis)| classify(identity, chassis),
        );
    }

    // Both power shelves expose no vendor and no system collection, so the
    // chassis is the only evidence there is.
    #[test]
    fn classifies_power_shelves_from_the_chassis_alone() {
        check_values(
            [
                Check {
                    scenario: "Lite-On, generic powershelf chassis id",
                    input: ChassisIdentity {
                        id: "powershelf",
                        manufacturer: Some("Lite-On"),
                        model: None,
                    },
                    expect: Some(HwType::LiteonPowerShelf),
                },
                Check {
                    scenario: "Lite-On, manufacturer on the chassis id",
                    input: ChassisIdentity {
                        id: "chassis",
                        manufacturer: Some("Lite-On Technology"),
                        model: None,
                    },
                    expect: Some(HwType::LiteonPowerShelf),
                },
                // Delta shares the generic "powershelf" id with Lite-On, and
                // the Lite-On arm claims that id unconditionally -- so Delta is
                // only reachable through the "chassis" id.
                Check {
                    scenario: "Delta, manufacturer on the chassis id",
                    input: ChassisIdentity {
                        id: "chassis",
                        manufacturer: Some("Delta Energy Systems"),
                        model: None,
                    },
                    expect: Some(HwType::DeltaPowerShelf),
                },
                Check {
                    scenario: "a non-power-shelf chassis is not a shelf",
                    input: ChassisIdentity {
                        id: "Card1",
                        manufacturer: Some("Delta"),
                        model: None,
                    },
                    expect: None,
                },
            ],
            |chassis| classify(ServiceIdentity::default(), &[chassis]),
        );
    }

    // is_delta_powershelf_chassis gates Delta detection: a power-shelf chassis
    // id ("chassis"/"powershelf") AND a Delta manufacturer. The manufacturer
    // check is case-insensitive and substring-based, and is what separates a
    // Delta shelf from a Lite-On shelf sharing the "powershelf" chassis id.
    #[test]
    fn is_delta_powershelf_chassis_gates_on_id_and_manufacturer() {
        let cases: [(&str, Option<&str>, bool); 9] = [
            // Delta manufacturer on either accepted power-shelf chassis id.
            ("chassis", Some("DELTA"), true),
            ("powershelf", Some("Delta"), true),
            // Case-insensitive, substring match on the manufacturer.
            ("chassis", Some("delta electronics"), true),
            ("powershelf", Some("Delta Energy Systems"), true),
            // Right manufacturer but a non-power-shelf chassis id is ignored.
            ("Card1", Some("DELTA"), false),
            ("Baseboard", Some("delta"), false),
            // Power-shelf chassis id but a different (or missing) manufacturer.
            ("powershelf", Some("Lite-On"), false),
            ("chassis", Some("NVIDIA"), false),
            ("chassis", None, false),
        ];
        for (id, mfg, expected) in cases {
            assert_eq!(
                is_delta_powershelf_chassis(id, mfg),
                expected,
                "id={id:?} manufacturer={mfg:?}"
            );
        }
    }

    // Lenovo's two variants differ only by the OEM identifier.
    #[test]
    fn lenovo_variants_split_on_the_oem_id() {
        check_values(
            [
                Check {
                    scenario: "Lenovo XCC",
                    input: None,
                    expect: Some(HwType::Lenovo),
                },
                Check {
                    scenario: "Lenovo with an AMI OEM block",
                    input: Some("Ami"),
                    expect: Some(HwType::LenovoAmi),
                },
            ],
            |oem_id| {
                classify(
                    ServiceIdentity {
                        vendor: Some("Lenovo"),
                        oem_id,
                        ..Default::default()
                    },
                    &[],
                )
            },
        );
    }

    // `needs_system_id` lets a caller that could not read the system collection
    // decide whether classifying anyway is safe. That promise only holds if the
    // vendors it clears really do classify identically either way.
    #[test]
    fn classification_ignores_system_id_for_other_vendors() {
        const VENDORS: [Option<&str>; 9] = [
            Some("AMI"),
            Some("Nvidia"),
            Some("NVIDIA"),
            Some("Dell"),
            Some("HPE"),
            Some("Lenovo"),
            Some("Supermicro"),
            Some("WIWYNN"),
            None,
        ];
        // Ids that change the answer for the vendors that do depend on one.
        const SYSTEM_IDS: [&str; 4] = ["DGX", "Bluefield", "BlueField_0", "system"];
        const PRODUCTS: [Option<&str>; 5] = [
            None,
            Some("GB BMC"),
            Some("GB200 NVL"),
            Some("VR NVL72"),
            Some("P3809"),
        ];

        for vendor in VENDORS {
            if needs_system_id(vendor) {
                continue;
            }
            for product in PRODUCTS {
                let without = classify(service(vendor, product), &[]);
                for system_id in SYSTEM_IDS {
                    let with = classify(
                        ServiceIdentity {
                            vendor,
                            product,
                            system_id: Some(system_id),
                            ..Default::default()
                        },
                        &[],
                    );
                    assert_eq!(
                        with, without,
                        "vendor={vendor:?} product={product:?} system_id={system_id:?}",
                    );
                }
            }
        }
    }

    // Some Supermicro BMCs name themselves only in the OEM block.
    #[test]
    fn supermicro_falls_back_to_the_oem_id_when_no_vendor_is_reported() {
        assert_eq!(
            classify(
                ServiceIdentity {
                    oem_id: Some("Supermicro"),
                    ..Default::default()
                },
                &[],
            ),
            Some(HwType::Supermicro),
        );
    }
}
