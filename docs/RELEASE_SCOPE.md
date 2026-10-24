# Initial public release scope

This release extracts and refactors selected inference and analysis operations
from the MSCL research project into a standalone package. It deliberately does
not publish the research workspace wholesale.

## Included implementation

The hierarchical inference module retains the existing encoder parameter names
and the local mean-initialised cluster / global WSI aggregation operations.
At inference all selected patches are visible; no random masking, reconstruction
decoder, contrastive loss, gradient update or training launcher is included.
Real-IHC global aggregation converts predicted local IHC tokens into a
predicted-IHC WSI representation.

The statistical and plotting utilities have configurable paths and strict
patient alignment. These replace private directory assumptions and historical
figure-specific annotations. They operate on supplied predictions and cannot
reproduce an unpublished model fitting process from features alone.

## Explicitly withheld

- Representation pretraining and all loss implementations.
- Supervised downstream fitting and hyperparameter-search code.
- Checkpoints, optimiser states, distributed launchers and training logs.
- Institutional cohort manifests and participant-level research results.
- Original clinical tables, report text, identifiers, images and embeddings.
- Manuscript drafts, unpublished result graphics, private audit documents,
  environment files and credentials.

No date or commitment for releasing the withheld materials is implied.
Public code availability is separate from model-weight and clinical-data
availability. A manuscript code-availability statement should make that scope
clear rather than describe this as release of the complete training pipeline.

## Limits

The package does not certify clinical labels, reconstruct raw WSIs, perform
cross-stain registration, or replace the upstream CONCH extractor. The inference
code supports the hierarchical model only, not older single-level architectures.
The provided examples are artificial and must not be reported as experimental
evidence. Use the same applicable analysis definitions and source versions when
attempting manuscript reproduction.

MIT covers this repository's source, not external models, data, trademarks,
third-party packages or clinical-use approvals. Do not upload participant data
or model checkpoints in issue reports or pull requests.
