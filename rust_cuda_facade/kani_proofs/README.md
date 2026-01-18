# GNN Facade CLI - Kani Formal Verification

**CISA Secure by Design Compliance - January 2026**

Bit-precise formal verification of memory safety for the Rust/CUDA FFI boundary.

## Test Summary

| Type | Count | Status |
|------|-------|--------|
| Unit Tests | 76 | ✓ PASS |
| Kani Proof Harnesses | 19 | ✓ PASS |
| **Total Verifications** | **95** | ✓ PASS |

## Quick Start

```bash
# Run unit tests
cargo test

# Run formal verification (requires Kani)
cargo kani
```

## Kani Installation (one-time)

```bash
cargo install --locked kani-verifier
kani setup
```

## Verified Properties

| Category | Unit Tests | Kani Proofs |
|----------|------------|-------------|
| Graph Construction | 4 | - |
| Node Feature Access | 12 | 3 |
| Edge Operations | 14 | 5 |
| Adjacency List | 9 | 3 |
| Node Mask Manager | 7 | 2 |
| Edge Mask Manager | 8 | 2 |
| Buffer Validator | 14 | 4 |
| Edge Features | 3 | - |
| Boundary Values | 3 | - |
| Stress Tests | 2 | - |

## What This Proves

1. **Panic-Free FFI Boundary** - Operations never trigger panics that escape to CUDA
2. **Buffer Overflow Protection** - Edge/node indexing never accesses out-of-bounds memory  
3. **Nondeterministic Safety** - Logic is safe for any possible input value
4. **Boundary Conditions** - MAX_NODES and MAX_EDGES limits properly enforced

## Author

Matthew Abbott <mattbachg@gmail.com>

## License

MIT License
