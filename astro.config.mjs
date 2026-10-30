// @ts-check
import { defineConfig } from 'astro/config';

import cloudflare from '@astrojs/cloudflare';

// https://astro.build/config
export default defineConfig({
  // The advisor runs on Cloudflare Workers so the Gemini API key stays server-side.
  output: 'server',
  adapter: cloudflare(),
});
