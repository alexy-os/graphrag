import { config as dotenvConfig } from 'dotenv';
import { z } from 'zod';

dotenvConfig();

const EnvSchema = z.object({
  QDRANT_URL: z.string().url(),
  QDRANT_KEY: z.string().min(1),

  NEO4J_URI: z.string().url(),
  NEO4J_USERNAME: z.string().min(1),
  NEO4J_PASSWORD: z.string().min(1),

  OPENROUTER_URL: z.string().url(),
  OPENROUTER_API_KEY: z.string().min(1),

  EMBEDDING_URL: z.string().url(),
  EMBEDDING_API_KEY: z.string().min(1),
});

const parsed = EnvSchema.safeParse(process.env);

if (!parsed.success) {
  console.error('Invalid environment variables:', parsed.error.flatten().fieldErrors);
  throw new Error('Environment validation failed');
}

export const ENV = parsed.data;


