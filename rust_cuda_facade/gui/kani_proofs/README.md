# GNN Facade - Kani Formal Verification

**CISA Secure by Design Compliance - January 2026**

Bit-precise formal verification of memory safety for the FFI boundary between Rust and Qt/CUDA layers.

## Quick Start

```bash
# Install Kani (one-time)
cargo install --locked kani-verifier
kani setup

# Run verification
cargo kani
```

## Expected Output

```
Complete - 14 successfully verified harnesses, 0 failures, 14 total.
```

## Verified Properties

| Category | Proofs | Status |
|----------|--------|--------|
| Node Feature Access | 2 | ✓ PASS |
| Edge Bounds Checking | 3 | ✓ PASS |
| Adjacency List Safety | 1 | ✓ PASS |
| Node Mask Operations | 2 | ✓ PASS |
| Edge Mask Operations | 1 | ✓ PASS |
| Buffer Index Validation | 3 | ✓ PASS |
| Degree Calculations | 2 | ✓ PASS |

## What This Proves

1. **Panic-Free FFI Boundary** - Operations never trigger panics that escape to C++/CUDA
2. **Buffer Overflow Protection** - Edge/node indexing never accesses out-of-bounds memory
3. **Nondeterministic Safety** - Logic is safe for any possible input value

## Files

- `src/lib.rs` - Proof harnesses and verifiable graph structures
- `VERIFICATION_REPORT.md` - Full auditor report

## Author

Matthew Abbott <mattbachg@gmail.com>

## License

MIT License
