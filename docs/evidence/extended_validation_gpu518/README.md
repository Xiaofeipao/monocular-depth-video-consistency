# GPU 518 video validation

This contains numerical evidence for the same four self-recorded clips and two
TUM excerpts evaluated with metric VDA-Small, GPU FP32 and input size 518.
Postprocessing still runs on CPU. The audit verifies original RGB identity,
prediction hashes, common evaluation masks, paired CSV aggregates and all six
comparison videos. Software versions are included without personal machine
paths or hardware model names.

See [the protocol and reproduction instructions](../extended_validation_cpu/README.md),
[full results](../../extended_validation_gpu518.md), and
[the same-frame comparison](../../video_resolution_comparison.md).
The CPU reference and GPU run change input size and device together.

Do not omit the failed C boundary constraints on both TUM excerpts or the
negative Depth Pro boundary result in the processed iBims audit. D's favorable
pose error is conditional on these two excerpts from one sequence, without a
cross-sequence significance claim.
