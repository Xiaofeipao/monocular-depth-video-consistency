# Optimization evidence

`tum16` and `tokyo32` compare original and explicitly named optimized variants
on the same cached CUDA metric VDA predictions from `gpu_20261028`.
The source manifest hash is recorded in each JSON. These are real inputs,
not synthetic depth. Protocols and caveats are in
[the optimization review](../../part3_results.md).

TUM/Tokyo jobs overlapped: raw postprocess FPS and automatic FPS winner fields
are exploratory, subject to CPU contention, and must not support speed claims.
Numerical depth/temporal metrics remain valid; timings are retained unedited.
TUM's GT-defined pose metric provides a crosscheck independent of fusion flow.
Application GT is incomplete, so warning quality metrics are unavailable.
B and B+C+D were deliberately omitted because independent landmarks are absent.

Final verification: 100 tests passed; one optional official DepthPro boundary
test skipped because its external source is not installed. The new integration
test covers pose evaluation with a default depth scale and sparse GT.
