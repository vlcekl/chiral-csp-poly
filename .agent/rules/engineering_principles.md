---
description: Core engineering principles and best practices
---

# Engineering Principles

Strive for excellence in every code change.

1.  **DRY (Don't Repeat Yourself)**: Avoid duplication. Refactor repeated logic into helper functions or classes.
2.  **Modularity**: Build small, focused components. Follow the existing module structure.
3.  **Configurability**: Use Hydra (`configs/`). Avoid hardcoding magic numbers or paths.
4.  **Efficiency**: Write performant code. Avoid unnecessary loops or expensive operations in hot paths.
5.  **Instruction Transferability**: Rely on `docs/` for deep understanding ("Why"), but follow these rules for immediate execution ("What").
