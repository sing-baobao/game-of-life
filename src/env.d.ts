/// <reference types="astro/client" />

declare namespace Cloudflare {
	interface Env {
		/** Provided at runtime via a Cloudflare Worker secret (or `.dev.vars` locally). */
		GEMINI_API_KEY: string;
		/**
		 * Optional comma-separated allowlist of origins permitted to call the
		 * advisor cross-origin. When unset, any origin is allowed.
		 */
		ALLOWED_ORIGINS?: string;
	}
}
