use num_bigint::BigUint;

use std::ops::BitOr;
use std::ops::BitOrAssign;
use std::ops::BitAnd;
//use std::ops::BitAndAssign;
//use std::ops::BitXor;
use std::ops::BitXorAssign;
use std::ops::Shl;
use std::ops::Shr;
use std::ops::ShlAssign;
use std::ops::ShrAssign;
use std::ops::AddAssign;
use std::ops::SubAssign;
use std::ops::Sub;
use std::cmp::Ordering;
use std::fmt;

use pyo3::prelude::*;
use pyo3::PyResult;
use pyo3::types::{PyInt, PyBytes, PyAny};
use pyo3::conversion::{FromPyObject, IntoPy};
use std::convert::TryInto;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct U256(u128,u128);

// Zero

impl U256 {
    pub fn zero() -> U256 {
        U256(0, 0)
    }
}

// One

impl U256 {
    pub fn one() -> U256 {
        U256(0, 1)
    }
}

// From

macro_rules! impl_from_smaller_integers_for_u256 {
    ($($t:ty),+) => {
        $(
            impl From<$t> for U256 {
                fn from(value: $t) -> Self {
                    U256(0, value as u128)
                }
            }
        )+
    };
}

impl_from_smaller_integers_for_u256!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

// BitOr

impl BitOr for U256 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        U256(self.0 | rhs.0, self.1 | rhs.1)
    }
}

impl<'a> BitOr<&'a U256> for U256 {
    type Output = Self;
    fn bitor(self, rhs: &'a Self) -> Self::Output {
        U256(self.0 | rhs.0, self.1 | rhs.1)
    }
}

impl<'a> BitOr<U256> for &'a U256 {
    type Output = U256;
    fn bitor(self, rhs: U256) -> Self::Output {
        U256(self.0 | rhs.0, self.1 | rhs.1)
    }
}

impl<'a, 'b> BitOr<&'b U256> for &'a U256 {
    type Output = U256;
    fn bitor(self, rhs: &'b U256) -> Self::Output {
        U256(self.0 | rhs.0, self.1 | rhs.1)
    }
}

// BitOrAssign

impl BitOrAssign for U256 {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
        self.1 |= rhs.1;
    }
}

impl<'a> BitOrAssign<&'a U256> for U256 {
    fn bitor_assign(&mut self, rhs: &'a Self) {
        self.0 |= rhs.0;
        self.1 |= rhs.1;
    }
}

// BitAnd

impl BitAnd for U256 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        U256(self.0 & rhs.0, self.1 & rhs.1)
    }
}

impl<'a> BitAnd<&'a U256> for U256 {
    type Output = Self;
    fn bitand(self, rhs: &'a Self) -> Self::Output {
        U256(self.0 & rhs.0, self.1 & rhs.1)
    }
}

impl<'a> BitAnd<U256> for &'a U256 {
    type Output = U256;
    fn bitand(self, rhs: U256) -> Self::Output {
        U256(self.0 & rhs.0, self.1 & rhs.1)
    }
}

impl<'a, 'b> BitAnd<&'b U256> for &'a U256 {
    type Output = U256;
    fn bitand(self, rhs: &'b U256) -> Self::Output {
        U256(self.0 & rhs.0, self.1 & rhs.1)
    }
}

// BitAndAssign (NOT YET NEEDED)

// BitXor (NOT YET NEEDED)

// BitXorAssign

impl BitXorAssign for U256 {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
        self.1 ^= rhs.1;
    }
}

impl<'a> BitXorAssign<&'a U256> for U256 {
    fn bitxor_assign(&mut self, rhs: &'a Self) {
        self.0 ^= rhs.0;
        self.1 ^= rhs.1;
    }
}

// Shift left

impl Shl<usize> for U256 {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output {
        if rhs >= 256 {
            U256(0, 0)
        } else if rhs == 0 {
            self
        } else if rhs < 128 {
            // Shift where some bits will move from the lower to the upper
            let high = (self.0 << rhs) | (self.1 >> (128 - rhs));
            let low = self.1 << rhs;
            U256(high, low)
        } else {
            // Shift entirely into the upper, with the lower part becoming all zeros
            let high = self.1 << (rhs - 128);
            U256(high, 0)
        }
    }
}

impl<'a> Shl<usize> for &'a U256 {
    type Output = U256;

    fn shl(self, rhs: usize) -> Self::Output {
        if rhs >= 256 {
            U256(0, 0)
        } else if rhs == 0 {
            *self
        } else if rhs < 128 {
            let high = (self.0 << rhs) | (self.1 >> (128 - rhs));
            let low = self.1 << rhs;
            U256(high, low)
        } else {
            let high = self.1 << (rhs - 128);
            U256(high, 0)
        }
    }
}

// Shift right

impl Shr<usize> for U256 {
    type Output = Self;

    fn shr(self, rhs: usize) -> Self::Output {
        if rhs >= 256 {
            U256(0, 0)
        } else if rhs == 0 {
            self
        } else if rhs < 128 {
            // Shift where some bits will move from the upper to the lower
            let low = (self.1 >> rhs) | (self.0 << (128 - rhs));
            let high = self.0 >> rhs;
            U256(high, low)
        } else {
            // Shift entirely into the lower, with the upper part becoming all zeros
            let low = self.0 >> (rhs - 128);
            U256(0, low)
        }
    }
}

impl<'a> Shr<usize> for &'a U256 {
    type Output = U256;

    fn shr(self, rhs: usize) -> Self::Output {
        if rhs >= 256 {
            U256(0, 0)
        } else if rhs == 0 {
            *self
        } else if rhs < 128 {
            let low = (self.1 >> rhs) | (self.0 << (128 - rhs));
            let high = self.0 >> rhs;
            U256(high, low)
        } else {
            let low = self.0 >> (rhs - 128);
            U256(0, low)
        }
    }
}

// ShrAssign

impl ShrAssign<usize> for U256 {
    fn shr_assign(&mut self, rhs: usize) {
        if rhs >= 256 {
            // If shifting by 256 or more, the result is zero
            self.0 = 0;
            self.1 = 0;
        } else if rhs == 0 {
            return;
        } else if rhs == 128 {
            // If shifting by exactly 128, high becomes low, and low becomes zero
            self.1 = self.0;
            self.0 = 0;
        } else if rhs > 128 {
            // If shifting by more than 128, shift the high part right by (rhs - 128)
            // and set the low part to the result, high part becomes zero
            self.1 = self.0 >> (rhs - 128);
            self.0 = 0;
        } else {
            // For shifts of less than 128 bits, need to handle the carry from high to low
            self.1 = (self.0 << (128 - rhs)) | (self.1 >> rhs);
            self.0 >>= rhs;
        }
    }
}

// ShlAssign

impl ShlAssign<usize> for U256 {
    fn shl_assign(&mut self, rhs: usize) {
        if rhs >= 256 {
            self.0 = 0;
            self.1 = 0;
        } else if rhs == 0 {
            return;
        } else if rhs == 128 {
            self.0 = self.1;
            self.1 = 0;
        } else if rhs > 128 {
            self.0 = self.1 << (rhs - 128);
            self.1 = 0;
        } else {
            self.0 = (self.0 << rhs) | (self.1 >> (128 - rhs));
            self.1 <<= rhs;
        }
    }
}

// AddAssign

impl AddAssign for U256 {
    fn add_assign(&mut self, rhs: Self) {
        let (low, carry) = self.1.overflowing_add(rhs.1);
        self.1 = low;
        let (high_sum, overflow1) = self.0.overflowing_add(rhs.0);
        let (high_sum, overflow2) = high_sum.overflowing_add(carry as u128);
        self.0 = high_sum;
        let _ = overflow1 || overflow2;
    }
}

// SubAssign

impl SubAssign for U256 {
    fn sub_assign(&mut self, rhs: Self) {
        let (low, borrow) = self.1.overflowing_sub(rhs.1);
        self.1 = low;
        self.0 = self.0.wrapping_sub(rhs.0).wrapping_sub(borrow as u128);
    }
}

impl<'a> SubAssign<&'a U256> for U256 {
    fn sub_assign(&mut self, rhs: &'a Self) {
        let (low, borrow) = self.1.overflowing_sub(rhs.1);
        self.1 = low;
        self.0 = self.0.wrapping_sub(rhs.0).wrapping_sub(borrow as u128);
    }
}

// Sub

impl Sub for U256 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = self;
        out -= rhs;
        out
    }
}

impl<'a> Sub<&'a U256> for U256 {
    type Output = Self;
    fn sub(self, rhs: &'a Self) -> Self::Output {
        let mut out = self;
        out -= rhs;
        out
    }
}

// Ordering

impl Ord for U256 {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.0.cmp(&other.0) {
            Ordering::Equal => self.1.cmp(&other.1),
            other => other,
        }
    }
}

impl PartialOrd for U256 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Trailing zeros

impl U256 {
    pub fn trailing_zeros(&self) -> u32 {
        if self.1 != 0 {
            self.1.trailing_zeros()
        } else {
            128 + self.0.trailing_zeros()
        }
    }
}

// Various conversions

impl U256 {
    pub fn to_usize(&self) -> Option<usize> {
        if self.0 == 0 && self.1 <= usize::MAX as u128 {
            Some(self.1 as usize)
        } else {
            None
        }
    }

    pub fn to_be_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[..16].copy_from_slice(&self.0.to_be_bytes());
        bytes[16..].copy_from_slice(&self.1.to_be_bytes());
        bytes
    }

    pub fn from_be_bytes(bytes: [u8; 32]) -> Self {
        let high = u128::from_be_bytes(bytes[..16].try_into().unwrap());
        let low = u128::from_be_bytes(bytes[16..].try_into().unwrap());
        U256(high, low)
    }
}

// PyO3

impl<'source> FromPyObject<'source> for U256 {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        // Extract the PyInt
        let py_int: &PyInt = obj.extract()?;
        
        // Convert PyInt to bytes
        let big_int_bytes = py_int.to_object(obj.py()).extract::<&PyBytes>(obj.py())?.as_bytes().to_vec();

        // Since PyBytes gives us a Vec<u8>, we can now use it to create a BigUint
        let val = BigUint::from_bytes_be(&big_int_bytes);

        // Now that we have BigUint, we can convert it to a U256
        // Ensure we pad the byte array to fit into a U256
        let mut padded_blob = [0u8; 32];
        let bytes = val.to_bytes_be(); // Convert BigUint back to bytes
        let padding_needed = 32usize.saturating_sub(bytes.len());
        padded_blob[padding_needed..].copy_from_slice(&bytes);

        // Construct U256 from padded_blob
        let (high_bytes, low_bytes) = padded_blob.split_at(16);
        let high = u128::from_be_bytes(high_bytes.try_into().unwrap());
        let low = u128::from_be_bytes(low_bytes.try_into().unwrap());

        Ok(U256(high, low))
    }
}

impl IntoPy<PyObject> for U256 {
    fn into_py(self, py: Python) -> PyObject {
        let bytes = [
            &self.0.to_be_bytes()[..],
            &self.1.to_be_bytes()[..],
        ]
        .concat();
        PyBytes::new(py, &bytes).into()
    }
}


impl U256 {
    pub fn from_u64_vec(vec: Vec<u64>) -> Self {
        let mut iter = vec.into_iter().rev(); // Reverse to work in little-endian order

        // Initialize parts to 0
        let mut high: u128 = 0;
        let mut low: u128 = 0;

        // Extract elements from the iterator, treating missing elements as 0
        let parts: Vec<u64> = iter.by_ref().take(4).collect();

        // Depending on the number of elements, fill the high and low parts
        for (i, &part) in parts.iter().enumerate() {
            if i < 2 {
                // First two elements go into the low part
                low |= (part as u128) << (64 * i);
            } else {
                // Next two elements go into the high part
                high |= (part as u128) << (64 * (i - 2));
            }
        }

        U256(high, low)
    }
}

impl U256 {
    pub fn from_u128_vec(vec: Vec<u128>) -> Self {
        match vec.len() {
            0 => U256(0, 0), // No elements: return U256 with both parts as 0
            1 => U256(0, vec[0]), // Single element: treat as low part, high part is 0
            // Two or more elements: first element is low, second is high, others are ignored
            _ => U256(vec[1], vec[0]),
        }
    }
    pub fn count_ones(&self) -> u32 {
        self.0.count_ones() + self.1.count_ones()
    }
}

// DISPLAYING

impl fmt::Display for U256 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:x}::{:x}", self.0, self.1)
    }
}
