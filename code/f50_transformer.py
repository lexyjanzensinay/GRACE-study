"""Fixed F50 algebraic reduction; complete observations only."""
import numpy as np
from grace_core import _canonicalize_X,default_prefix_transformer
F50_INDICES=tuple(i for i in range(78) if i%13<10 and i%13!=8 and not(i//13==2 and i%13 in (0,6,7,9)))
class F50Transformer:
 def __call__(self,X,tau):
  batch=_canonicalize_X(X)
  if batch.shape[2]!=6 or not np.all(batch.observation_mask[:,:tau,:]) or not np.isfinite(batch.values[:,:tau,:]).all():raise ValueError('F50 requires six fully observed finite channels; missing/padded support is not authorized')
  return default_prefix_transformer(batch,tau)[:,F50_INDICES]
