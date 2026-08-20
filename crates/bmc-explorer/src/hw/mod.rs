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

//! Per-platform BMC exploration helpers.
//!
//! The [`HwType`] taxonomy and the rules that resolve one from Redfish live in
//! the `hw-platform` crate, so `carbide-health` can classify the same way
//! without depending on this crate's exploration types. They are re-exported
//! here because every caller in this crate reaches them through `hw::`.

pub use hw_platform::{BiosAttr, BiosAttrValue, HwType};

pub mod bluefield;
pub mod dell;
pub mod gb200;
pub mod hpe;
pub mod lenovo;
pub mod lenovo_ami;
pub mod lenovo_gb300;
pub mod supermicro;
pub mod supermicro_gb300;
pub mod vera_rubin;
pub mod viking;
