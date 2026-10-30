/// <reference types="astro/client" />

declare namespace Cloudflare {
	interface Env {
		/** Provided at runtime via a Cloudflare Worker secret (or `.dev.vars` locally). */
		GEMINI_API_KEY: string;
	}
}
