use std::cmp::Ordering;
use std::fmt;
use std::hash::Hash;
use std::ops::{
    AddAssign, BitAnd, BitOr, BitOrAssign, BitXorAssign, Shl, ShlAssign, Shr, ShrAssign, Sub,
    SubAssign,
};

#[inline(always)]
fn add_with_carry_u128(lhs: u128, rhs: u128, carry: bool) -> (u128, bool) {
    let (sum, c1) = lhs.overflowing_add(rhs);
    let (sum, c2) = sum.overflowing_add(carry as u128);
    (sum, c1 || c2)
}

#[inline(always)]
fn sub_with_borrow_u128(lhs: u128, rhs: u128, borrow: bool) -> (u128, bool) {
    let (diff, b1) = lhs.overflowing_sub(rhs);
    let (diff, b2) = diff.overflowing_sub(borrow as u128);
    (diff, b1 || b2)
}

pub trait BitBackend:
    Copy
    + Clone
    + Eq
    + Ord
    + Hash
    + fmt::Display
    + Send
    + Sync
    + 'static
    + From<u8>
    + From<u128>
    + BitOr<Output = Self>
    + BitOrAssign<Self>
    + BitAnd<Output = Self>
    + BitXorAssign<Self>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
    + ShlAssign<usize>
    + ShrAssign<usize>
    + AddAssign<Self>
    + SubAssign<Self>
    + Sub<Output = Self>
{
    const LIMBS: usize;
    const BIT_WIDTH: usize;
    const FITS_U256: bool;

    fn zero() -> Self;
    fn one() -> Self;
    fn trailing_zeros(&self) -> u32;
    fn count_ones(&self) -> u32;
    fn to_usize(&self) -> Option<usize>;
    fn to_be_bytes_vec(&self) -> Vec<u8>;
    fn from_be_slice(bytes: &[u8]) -> Self;
    fn from_u128_slice(slice: &[u128]) -> Self;
    fn to_u256(&self) -> U256;
    fn from_u256(value: U256) -> Self;
}

impl BitBackend for u128 {
    const LIMBS: usize = 1;
    const BIT_WIDTH: usize = 128;
    const FITS_U256: bool = true;

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }

    #[inline]
    fn trailing_zeros(&self) -> u32 {
        u128::trailing_zeros(*self)
    }

    #[inline]
    fn count_ones(&self) -> u32 {
        u128::count_ones(*self)
    }

    #[inline]
    fn to_usize(&self) -> Option<usize> {
        if *self <= usize::MAX as u128 {
            Some(*self as usize)
        } else {
            None
        }
    }

    #[inline]
    fn to_be_bytes_vec(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    #[inline]
    fn from_be_slice(bytes: &[u8]) -> Self {
        let mut padded = [0u8; 16];
        if bytes.len() >= 16 {
            padded.copy_from_slice(&bytes[bytes.len() - 16..]);
        } else {
            padded[(16 - bytes.len())..].copy_from_slice(bytes);
        }
        u128::from_be_bytes(padded)
    }

    #[inline]
    fn from_u128_slice(slice: &[u128]) -> Self {
        if slice.len() > 1 {
            debug_assert!(slice[1..].iter().all(|&x| x == 0));
        }
        slice.first().copied().unwrap_or(0)
    }

    #[inline]
    fn to_u256(&self) -> U256 {
        U256(0, *self)
    }

    #[inline]
    fn from_u256(value: U256) -> Self {
        debug_assert!(value.0 == 0);
        value.1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct U256(pub(crate) u128, pub(crate) u128);

impl U256 {
    #[inline]
    pub fn zero() -> U256 {
        U256(0, 0)
    }

    #[inline]
    pub fn one() -> U256 {
        U256(0, 1)
    }

    #[inline]
    pub fn to_be_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[..16].copy_from_slice(&self.0.to_be_bytes());
        bytes[16..].copy_from_slice(&self.1.to_be_bytes());
        bytes
    }

    #[inline]
    pub fn from_be_bytes(bytes: [u8; 32]) -> Self {
        let high = u128::from_be_bytes(bytes[..16].try_into().unwrap());
        let low = u128::from_be_bytes(bytes[16..].try_into().unwrap());
        U256(high, low)
    }
}

macro_rules! impl_from_smaller_integers_for_u256 {
    ($($t:ty),+) => {
        $(
            impl From<$t> for U256 {
                #[inline]
                fn from(value: $t) -> Self {
                    U256(0, value as u128)
                }
            }
        )+
    };
}

impl_from_smaller_integers_for_u256!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

impl BitOr for U256 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        U256(self.0 | rhs.0, self.1 | rhs.1)
    }
}

impl<'a> BitOr<&'a U256> for U256 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: &'a Self) -> Self::Output {
        U256(self.0 | rhs.0, self.1 | rhs.1)
    }
}

impl<'a> BitOr<U256> for &'a U256 {
    type Output = U256;

    #[inline]
    fn bitor(self, rhs: U256) -> Self::Output {
        U256(self.0 | rhs.0, self.1 | rhs.1)
    }
}

impl<'a, 'b> BitOr<&'b U256> for &'a U256 {
    type Output = U256;

    #[inline]
    fn bitor(self, rhs: &'b U256) -> Self::Output {
        U256(self.0 | rhs.0, self.1 | rhs.1)
    }
}

impl BitOrAssign for U256 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
        self.1 |= rhs.1;
    }
}

impl<'a> BitOrAssign<&'a U256> for U256 {
    #[inline]
    fn bitor_assign(&mut self, rhs: &'a Self) {
        self.0 |= rhs.0;
        self.1 |= rhs.1;
    }
}

impl BitAnd for U256 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        U256(self.0 & rhs.0, self.1 & rhs.1)
    }
}

impl<'a> BitAnd<&'a U256> for U256 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: &'a Self) -> Self::Output {
        U256(self.0 & rhs.0, self.1 & rhs.1)
    }
}

impl<'a> BitAnd<U256> for &'a U256 {
    type Output = U256;

    #[inline]
    fn bitand(self, rhs: U256) -> Self::Output {
        U256(self.0 & rhs.0, self.1 & rhs.1)
    }
}

impl<'a, 'b> BitAnd<&'b U256> for &'a U256 {
    type Output = U256;

    #[inline]
    fn bitand(self, rhs: &'b U256) -> Self::Output {
        U256(self.0 & rhs.0, self.1 & rhs.1)
    }
}

impl BitXorAssign for U256 {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
        self.1 ^= rhs.1;
    }
}

impl<'a> BitXorAssign<&'a U256> for U256 {
    #[inline]
    fn bitxor_assign(&mut self, rhs: &'a Self) {
        self.0 ^= rhs.0;
        self.1 ^= rhs.1;
    }
}

impl Shl<usize> for U256 {
    type Output = Self;

    #[inline]
    fn shl(self, rhs: usize) -> Self::Output {
        if rhs >= 256 {
            U256::zero()
        } else if rhs == 0 {
            self
        } else if rhs < 128 {
            let high = (self.0 << rhs) | (self.1 >> (128 - rhs));
            let low = self.1 << rhs;
            U256(high, low)
        } else if rhs == 128 {
            U256(self.1, 0)
        } else {
            U256(self.1 << (rhs - 128), 0)
        }
    }
}

impl<'a> Shl<usize> for &'a U256 {
    type Output = U256;

    #[inline]
    fn shl(self, rhs: usize) -> Self::Output {
        (*self) << rhs
    }
}

impl Shr<usize> for U256 {
    type Output = Self;

    #[inline]
    fn shr(self, rhs: usize) -> Self::Output {
        if rhs >= 256 {
            U256::zero()
        } else if rhs == 0 {
            self
        } else if rhs < 128 {
            let low = (self.1 >> rhs) | (self.0 << (128 - rhs));
            let high = self.0 >> rhs;
            U256(high, low)
        } else if rhs == 128 {
            U256(0, self.0)
        } else {
            U256(0, self.0 >> (rhs - 128))
        }
    }
}

impl<'a> Shr<usize> for &'a U256 {
    type Output = U256;

    #[inline]
    fn shr(self, rhs: usize) -> Self::Output {
        (*self) >> rhs
    }
}

impl ShrAssign<usize> for U256 {
    #[inline]
    fn shr_assign(&mut self, rhs: usize) {
        *self = *self >> rhs;
    }
}

impl ShlAssign<usize> for U256 {
    #[inline]
    fn shl_assign(&mut self, rhs: usize) {
        *self = *self << rhs;
    }
}

impl AddAssign for U256 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        let (low, carry) = add_with_carry_u128(self.1, rhs.1, false);
        let (high, _) = add_with_carry_u128(self.0, rhs.0, carry);
        self.0 = high;
        self.1 = low;
    }
}

impl<'a> AddAssign<&'a U256> for U256 {
    #[inline]
    fn add_assign(&mut self, rhs: &'a Self) {
        *self += *rhs;
    }
}

impl SubAssign for U256 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        let (low, borrow) = sub_with_borrow_u128(self.1, rhs.1, false);
        let (high, _) = sub_with_borrow_u128(self.0, rhs.0, borrow);
        self.0 = high;
        self.1 = low;
    }
}

impl<'a> SubAssign<&'a U256> for U256 {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a Self) {
        *self -= *rhs;
    }
}

impl Sub for U256 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = self;
        out -= rhs;
        out
    }
}

impl<'a> Sub<&'a U256> for U256 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        let mut out = self;
        out -= *rhs;
        out
    }
}

impl Ord for U256 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.0.cmp(&other.0) {
            Ordering::Equal => self.1.cmp(&other.1),
            non_eq => non_eq,
        }
    }
}

impl PartialOrd for U256 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for U256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:x}::{:x}", self.0, self.1)
    }
}

impl BitBackend for U256 {
    const LIMBS: usize = 2;
    const BIT_WIDTH: usize = 256;
    const FITS_U256: bool = true;

    #[inline]
    fn zero() -> Self {
        U256::zero()
    }

    #[inline]
    fn one() -> Self {
        U256::one()
    }

    #[inline]
    fn trailing_zeros(&self) -> u32 {
        if self.1 != 0 {
            self.1.trailing_zeros()
        } else {
            128 + self.0.trailing_zeros()
        }
    }

    #[inline]
    fn count_ones(&self) -> u32 {
        self.0.count_ones() + self.1.count_ones()
    }

    #[inline]
    fn to_usize(&self) -> Option<usize> {
        if self.0 == 0 && self.1 <= usize::MAX as u128 {
            Some(self.1 as usize)
        } else {
            None
        }
    }

    #[inline]
    fn to_be_bytes_vec(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    #[inline]
    fn from_be_slice(bytes: &[u8]) -> Self {
        let mut padded = [0u8; 32];
        if bytes.len() >= 32 {
            padded.copy_from_slice(&bytes[bytes.len() - 32..]);
        } else {
            padded[(32 - bytes.len())..].copy_from_slice(bytes);
        }
        U256::from_be_bytes(padded)
    }

    #[inline]
    fn from_u128_slice(slice: &[u128]) -> Self {
        match slice.len() {
            0 => U256(0, 0),
            1 => U256(0, slice[0]),
            _ => U256(slice[1], slice[0]),
        }
    }

    #[inline]
    fn to_u256(&self) -> U256 {
        *self
    }

    #[inline]
    fn from_u256(value: U256) -> Self {
        value
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct U768(
    pub(crate) u128,
    pub(crate) u128,
    pub(crate) u128,
    pub(crate) u128,
    pub(crate) u128,
    pub(crate) u128,
);

impl U768 {
    #[inline]
    pub fn zero() -> U768 {
        U768(0, 0, 0, 0, 0, 0)
    }

    #[inline]
    pub fn one() -> U768 {
        U768(0, 0, 0, 0, 0, 1)
    }

    #[inline]
    fn to_limbs_le(self) -> [u128; 6] {
        [self.5, self.4, self.3, self.2, self.1, self.0]
    }

    #[inline]
    fn from_limbs_le(limbs: [u128; 6]) -> Self {
        U768(limbs[5], limbs[4], limbs[3], limbs[2], limbs[1], limbs[0])
    }

    #[inline]
    pub fn to_be_bytes(&self) -> [u8; 96] {
        let mut bytes = [0u8; 96];
        bytes[0..16].copy_from_slice(&self.0.to_be_bytes());
        bytes[16..32].copy_from_slice(&self.1.to_be_bytes());
        bytes[32..48].copy_from_slice(&self.2.to_be_bytes());
        bytes[48..64].copy_from_slice(&self.3.to_be_bytes());
        bytes[64..80].copy_from_slice(&self.4.to_be_bytes());
        bytes[80..96].copy_from_slice(&self.5.to_be_bytes());
        bytes
    }

    #[inline]
    pub fn from_be_bytes(bytes: [u8; 96]) -> Self {
        U768(
            u128::from_be_bytes(bytes[0..16].try_into().unwrap()),
            u128::from_be_bytes(bytes[16..32].try_into().unwrap()),
            u128::from_be_bytes(bytes[32..48].try_into().unwrap()),
            u128::from_be_bytes(bytes[48..64].try_into().unwrap()),
            u128::from_be_bytes(bytes[64..80].try_into().unwrap()),
            u128::from_be_bytes(bytes[80..96].try_into().unwrap()),
        )
    }
}

macro_rules! impl_from_smaller_integers_for_u768 {
    ($($t:ty),+) => {
        $(
            impl From<$t> for U768 {
                #[inline]
                fn from(value: $t) -> Self {
                    U768(0, 0, 0, 0, 0, value as u128)
                }
            }
        )+
    };
}

impl_from_smaller_integers_for_u768!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

impl BitOr for U768 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        U768(
            self.0 | rhs.0,
            self.1 | rhs.1,
            self.2 | rhs.2,
            self.3 | rhs.3,
            self.4 | rhs.4,
            self.5 | rhs.5,
        )
    }
}

impl<'a> BitOr<&'a U768> for U768 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: &'a Self) -> Self::Output {
        self | *rhs
    }
}

impl<'a> BitOr<U768> for &'a U768 {
    type Output = U768;

    #[inline]
    fn bitor(self, rhs: U768) -> Self::Output {
        *self | rhs
    }
}

impl<'a, 'b> BitOr<&'b U768> for &'a U768 {
    type Output = U768;

    #[inline]
    fn bitor(self, rhs: &'b U768) -> Self::Output {
        *self | *rhs
    }
}

impl BitOrAssign for U768 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
        self.1 |= rhs.1;
        self.2 |= rhs.2;
        self.3 |= rhs.3;
        self.4 |= rhs.4;
        self.5 |= rhs.5;
    }
}

impl<'a> BitOrAssign<&'a U768> for U768 {
    #[inline]
    fn bitor_assign(&mut self, rhs: &'a Self) {
        *self |= *rhs;
    }
}

impl BitAnd for U768 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        U768(
            self.0 & rhs.0,
            self.1 & rhs.1,
            self.2 & rhs.2,
            self.3 & rhs.3,
            self.4 & rhs.4,
            self.5 & rhs.5,
        )
    }
}

impl<'a> BitAnd<&'a U768> for U768 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: &'a Self) -> Self::Output {
        self & *rhs
    }
}

impl<'a> BitAnd<U768> for &'a U768 {
    type Output = U768;

    #[inline]
    fn bitand(self, rhs: U768) -> Self::Output {
        *self & rhs
    }
}

impl<'a, 'b> BitAnd<&'b U768> for &'a U768 {
    type Output = U768;

    #[inline]
    fn bitand(self, rhs: &'b U768) -> Self::Output {
        *self & *rhs
    }
}

impl BitXorAssign for U768 {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
        self.1 ^= rhs.1;
        self.2 ^= rhs.2;
        self.3 ^= rhs.3;
        self.4 ^= rhs.4;
        self.5 ^= rhs.5;
    }
}

impl<'a> BitXorAssign<&'a U768> for U768 {
    #[inline]
    fn bitxor_assign(&mut self, rhs: &'a Self) {
        *self ^= *rhs;
    }
}

impl Shl<usize> for U768 {
    type Output = Self;

    #[inline]
    fn shl(self, rhs: usize) -> Self::Output {
        if rhs >= 768 {
            return U768::zero();
        }
        if rhs == 0 {
            return self;
        }
        let limb_shift = rhs / 128;
        let bit_shift = rhs % 128;
        let src = self.to_limbs_le();
        let mut out = [0u128; 6];
        if bit_shift == 0 {
            for dst in (limb_shift..6).rev() {
                out[dst] = src[dst - limb_shift];
            }
        } else {
            for dst in (0..6).rev() {
                if dst >= limb_shift {
                    let src_idx = dst - limb_shift;
                    out[dst] |= src[src_idx] << bit_shift;
                    if src_idx > 0 {
                        out[dst] |= src[src_idx - 1] >> (128 - bit_shift);
                    }
                }
            }
        }
        U768::from_limbs_le(out)
    }
}

impl<'a> Shl<usize> for &'a U768 {
    type Output = U768;

    #[inline]
    fn shl(self, rhs: usize) -> Self::Output {
        (*self) << rhs
    }
}

impl Shr<usize> for U768 {
    type Output = Self;

    #[inline]
    fn shr(self, rhs: usize) -> Self::Output {
        if rhs >= 768 {
            return U768::zero();
        }
        if rhs == 0 {
            return self;
        }
        let limb_shift = rhs / 128;
        let bit_shift = rhs % 128;
        let src = self.to_limbs_le();
        let mut out = [0u128; 6];
        if bit_shift == 0 {
            for dst in 0..(6 - limb_shift) {
                out[dst] = src[dst + limb_shift];
            }
        } else {
            for dst in 0..6 {
                let src_idx = dst + limb_shift;
                if src_idx < 6 {
                    out[dst] |= src[src_idx] >> bit_shift;
                    if src_idx + 1 < 6 {
                        out[dst] |= src[src_idx + 1] << (128 - bit_shift);
                    }
                }
            }
        }
        U768::from_limbs_le(out)
    }
}

impl<'a> Shr<usize> for &'a U768 {
    type Output = U768;

    #[inline]
    fn shr(self, rhs: usize) -> Self::Output {
        (*self) >> rhs
    }
}

impl ShrAssign<usize> for U768 {
    #[inline]
    fn shr_assign(&mut self, rhs: usize) {
        *self = *self >> rhs;
    }
}

impl ShlAssign<usize> for U768 {
    #[inline]
    fn shl_assign(&mut self, rhs: usize) {
        *self = *self << rhs;
    }
}

impl AddAssign for U768 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        let (v5, c5) = add_with_carry_u128(self.5, rhs.5, false);
        let (v4, c4) = add_with_carry_u128(self.4, rhs.4, c5);
        let (v3, c3) = add_with_carry_u128(self.3, rhs.3, c4);
        let (v2, c2) = add_with_carry_u128(self.2, rhs.2, c3);
        let (v1, c1) = add_with_carry_u128(self.1, rhs.1, c2);
        let (v0, _) = add_with_carry_u128(self.0, rhs.0, c1);
        self.0 = v0;
        self.1 = v1;
        self.2 = v2;
        self.3 = v3;
        self.4 = v4;
        self.5 = v5;
    }
}

impl<'a> AddAssign<&'a U768> for U768 {
    #[inline]
    fn add_assign(&mut self, rhs: &'a Self) {
        *self += *rhs;
    }
}

impl SubAssign for U768 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        let (v5, b5) = sub_with_borrow_u128(self.5, rhs.5, false);
        let (v4, b4) = sub_with_borrow_u128(self.4, rhs.4, b5);
        let (v3, b3) = sub_with_borrow_u128(self.3, rhs.3, b4);
        let (v2, b2) = sub_with_borrow_u128(self.2, rhs.2, b3);
        let (v1, b1) = sub_with_borrow_u128(self.1, rhs.1, b2);
        let (v0, _) = sub_with_borrow_u128(self.0, rhs.0, b1);
        self.0 = v0;
        self.1 = v1;
        self.2 = v2;
        self.3 = v3;
        self.4 = v4;
        self.5 = v5;
    }
}

impl<'a> SubAssign<&'a U768> for U768 {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a Self) {
        *self -= *rhs;
    }
}

impl Sub for U768 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = self;
        out -= rhs;
        out
    }
}

impl<'a> Sub<&'a U768> for U768 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        let mut out = self;
        out -= *rhs;
        out
    }
}

impl Ord for U768 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .cmp(&other.0)
            .then_with(|| self.1.cmp(&other.1))
            .then_with(|| self.2.cmp(&other.2))
            .then_with(|| self.3.cmp(&other.3))
            .then_with(|| self.4.cmp(&other.4))
            .then_with(|| self.5.cmp(&other.5))
    }
}

impl PartialOrd for U768 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for U768 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:x}::{:x}::{:x}::{:x}::{:x}::{:x}",
            self.0, self.1, self.2, self.3, self.4, self.5
        )
    }
}

impl BitBackend for U768 {
    const LIMBS: usize = 6;
    const BIT_WIDTH: usize = 768;
    const FITS_U256: bool = false;

    #[inline]
    fn zero() -> Self {
        U768::zero()
    }

    #[inline]
    fn one() -> Self {
        U768::one()
    }

    #[inline]
    fn trailing_zeros(&self) -> u32 {
        if self.5 != 0 {
            self.5.trailing_zeros()
        } else if self.4 != 0 {
            128 + self.4.trailing_zeros()
        } else if self.3 != 0 {
            256 + self.3.trailing_zeros()
        } else if self.2 != 0 {
            384 + self.2.trailing_zeros()
        } else if self.1 != 0 {
            512 + self.1.trailing_zeros()
        } else {
            640 + self.0.trailing_zeros()
        }
    }

    #[inline]
    fn count_ones(&self) -> u32 {
        self.0.count_ones()
            + self.1.count_ones()
            + self.2.count_ones()
            + self.3.count_ones()
            + self.4.count_ones()
            + self.5.count_ones()
    }

    #[inline]
    fn to_usize(&self) -> Option<usize> {
        if self.0 == 0
            && self.1 == 0
            && self.2 == 0
            && self.3 == 0
            && self.4 == 0
            && self.5 <= usize::MAX as u128
        {
            Some(self.5 as usize)
        } else {
            None
        }
    }

    #[inline]
    fn to_be_bytes_vec(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    #[inline]
    fn from_be_slice(bytes: &[u8]) -> Self {
        let mut padded = [0u8; 96];
        if bytes.len() >= 96 {
            padded.copy_from_slice(&bytes[bytes.len() - 96..]);
        } else {
            padded[(96 - bytes.len())..].copy_from_slice(bytes);
        }
        U768::from_be_bytes(padded)
    }

    #[inline]
    fn from_u128_slice(slice: &[u128]) -> Self {
        let mut limbs = [0u128; 6];
        let count = slice.len().min(6);
        limbs[..count].copy_from_slice(&slice[..count]);
        U768::from_limbs_le(limbs)
    }

    #[inline]
    fn to_u256(&self) -> U256 {
        assert!(
            self.0 == 0 && self.1 == 0 && self.2 == 0 && self.3 == 0,
            "attempted to truncate U768 with nonzero high limbs into U256"
        );
        U256(self.4, self.5)
    }

    #[inline]
    fn from_u256(value: U256) -> Self {
        U768(0, 0, 0, 0, value.0, value.1)
    }
}
