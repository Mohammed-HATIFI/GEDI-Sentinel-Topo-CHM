# Provenance policy

Every reported number must resolve to a frozen source table, support definition, generating code/notebook, checkpoint SHA-256, and input-product/version manifest.

Absolute paths are not provenance. A release entry needs a logical ID, relative archive path, checksum, byte size, source URL/DOI, processing operator, CRS/grid, and NoData convention.

`configs/phase1_models.json` and `configs/final_models.json` are the portable, path-free registries. The exact research-code snapshot can retain historical execution paths, but public configuration files must not contain local drive letters.

Never infer missing hashes, versions, licences, or archive URLs. Keep them as explicit release gates until verified.
