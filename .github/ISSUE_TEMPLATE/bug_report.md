---
name: Bug report
about: A fit fails, gives a wrong result, or the code crashes
labels: bug
---

**What happened**
A short description of the problem, and what you expected instead.

**How to reproduce**
A minimal script, ideally with a small curve (or attach it as a CSV):

```python
import numpy as np
import fitpo

curve = np.loadtxt("curve.csv", delimiter=",")
res = fitpo.fit_deluca_backbone(curve)
```

**Error message or output**
Paste the full traceback or the result.

**Versions**
- fitpo: (`python -c "import fitpo; print(fitpo.__version__)"`)
- Python:
- numpy / scipy:
- Operating system:
