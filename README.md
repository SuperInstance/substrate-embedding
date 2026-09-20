# substrate-embedding

BGE-Large-compatible embedding client with cache + JEV gate.

```typescript
import { EmbeddingClient } from 'substrate-embedding';

const client = new EmbeddingClient();
const v1 = await client.embed('cell-witness');
const v2 = await client.embed('cell-bind');
client.similarity(v1, v2);  // 0-1
client.jevScore(v1);  // 0.65-0.95
```
