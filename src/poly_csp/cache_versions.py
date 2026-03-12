"""Central cache invalidation boundaries for poly_csp.

`*_SCHEMA_VERSION` is for on-disk payload shape and serialization contract changes.
`*_MODEL_VERSION` is for scientific/chemical behavior changes that alter cached content
without necessarily changing the JSON layout.
"""

RUNTIME_PAYLOAD_CACHE_SCHEMA_VERSION = 4
RUNTIME_PAYLOAD_MODEL_VERSION = 1

BACKBONE_POSE_CACHE_SCHEMA_VERSION = 2
BACKBONE_POSE_MODEL_VERSION = 1
