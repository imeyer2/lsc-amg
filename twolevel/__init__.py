"""Two-level learned vs classical strength-of-connection study (October 2026).

Modules:
  common     paths ($LSC_OUTPUTS/twolevel), matrix sets (test seed 1000, tuning seed 2000, ...)
  rules      classical rules theta / sym / topk / srowq and C = max(S, S^T) + I
  harness    the timing harness (all methods: classical, paper GAT policies, lin, node, two, mask)
  rowthr     per-row threshold model 'lin', hand rule 'two', sep-CMA, measured-time evaluator
  nodepol    node-modulated policy 'node' (noderow / nodeabs) and the two-level cost model
  train_lin  CMA training of lin on measured time;  train_node  ES training of node on the cost model
  costmodel  classical sweeps + cost-model fit;  oracle  per-matrix headroom search + mask export
  profiling  inference / setup cost breakdowns;  tables  tables, paired tests, winners, headroom, LU fit
Driver: scripts/twolevel.py. Results and findings: results/twolevel_2026-10/README.md.
"""
