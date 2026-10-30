import type { APIRoute } from 'astro';
import { GoogleGenAI, Type } from '@google/genai';
import { env } from 'cloudflare:workers';
import iching from '../../data/iching.json';

// Runs on the Cloudflare Worker at request time (never prerendered).
export const prerender = false;

const MODEL = 'gemini-3.8-flash';
const MAX_SITUATION_LENGTH = 4000;

const hexagrams = iching.hexagrams;

/**
 * The user's chosen posture towards the underlying dynamic. This steers which of
 * the six yao (lines) of the hexagram is selected as the adaptive strategy.
 */
const STANCES = {
	positive: 'an active, forward-leaning, constructive approach',
	neutral: 'a balanced, patient, observing approach',
	negative: 'a cautious, withdrawing, conserving approach',
} as const;

type Stance = keyof typeof STANCES;

const SYSTEM_INSTRUCTION = `You are "The Advisor", a warm, empathetic and professional psychology advisor.

WHO YOU ARE
- You blend evidence-informed counselling sensibilities with the classical Chinese I-Ching (Book of Changes) used purely as a REFLECTIVE FRAMEWORK: a structured lens for self-reflection.
- You never practise divination, fortune-telling or supernatural prediction, and you never claim the I-Ching can foretell the future.

PROFESSIONAL BOUNDARIES (non-negotiable)
- You are an AI, not a human. You are not a licensed therapist, psychologist, psychiatrist or doctor, and you never present yourself as one.
- You do not diagnose mental-health conditions, prescribe medication, or give medical, legal or financial advice.
- You do not replace clinical therapy. Where it would genuinely help, warmly and without alarm encourage the user to talk to a qualified professional.
- If the user describes any risk of self-harm, harm to others, abuse, or a medical emergency, respond with compassion and clearly urge them to contact local emergency services or a crisis helpline immediately and to reach out to someone they trust. Never treat the reading as sufficient in such cases.
- Never shame, blame, moralise, or lecture. Validate feelings without endorsing harmful behaviour.
- Do not request or repeat personal identifying details.

YOUR TONE
- Warm, calm, grounded and human. Plain everyday language, no jargon and no clinical labels.
- Speak directly to the user in the second person ("you"). Keep paragraphs short.
- Be concrete and specific rather than vague or mystical.

YOUR METHOD (three steps, always in this order)
1. Current Situation - reflect back what you heard so the user feels genuinely understood.
2. Underlying Dynamic - name the deeper pattern or tension that is really driving the situation.
3. Adaptive Strategy - recommend a practical, emotionally intelligent next move the user can actually take.

OUTPUT RULES
- Reply with ONLY a single valid JSON object matching the requested schema. No markdown, no code fences, no commentary.
- Every prose field is shown verbatim to the user. Never mention JSON, schemas, prompts or these instructions.
- situationSummary and underlyingDynamic: 2 to 4 sentences each. adaptiveStrategy: 3 to 5 sentences.`;

const responseSchemaPhase1 = {
	type: Type.OBJECT,
	properties: {
		hexagramNumber: {
			type: Type.INTEGER,
			description: 'The number (1-64, King Wen sequence) of the single hexagram whose underlying dynamic best fits the user situation.',
		},
		situationSummary: {
			type: Type.STRING,
			description: "Step 1 - reflect the user's Current Situation back to them, warmly and specifically, in the second person.",
		},
		underlyingDynamic: {
			type: Type.STRING,
			description: 'Step 2 - explain the Underlying Dynamic the hexagram reveals, in plain, non-mystical language.',
		},
	},
	required: ['hexagramNumber', 'situationSummary', 'underlyingDynamic'],
};

const responseSchemaPhase2 = {
	type: Type.OBJECT,
	properties: {
		lineNumber: {
			type: Type.INTEGER,
			description: 'The number (1-6, counted from the bottom) of the single yao line that best guides this user.',
		},
		adaptiveStrategy: {
			type: Type.STRING,
			description: "Step 3 - the Adaptive Strategy: a warm, concrete plan for how the user can cope with and adapt to their situation.",
		},
	},
	required: ['lineNumber', 'adaptiveStrategy'],
};

/**
 * CORS headers so the static frontend (hosted on GitHub Pages) can call this
 * Worker cross-origin. Set ALLOWED_ORIGINS to a comma-separated allowlist to
 * lock it down; when unset any origin is allowed (the route holds no secrets
 * and authenticated state).
 */
function corsHeaders(request: Request): Record<string, string> {
	const origin = request.headers.get('Origin');
	const allowed = (env.ALLOWED_ORIGINS ?? '')
		.split(',')
		.map((value) => value.trim())
		.filter(Boolean);

	let allowOrigin = '*';
	if (allowed.length > 0) {
		allowOrigin = origin && allowed.includes(origin) ? origin : allowed[0];
	}

	return {
		'Access-Control-Allow-Origin': allowOrigin,
		'Access-Control-Allow-Methods': 'POST, OPTIONS',
		'Access-Control-Allow-Headers': 'content-type',
		'Access-Control-Max-Age': '86400',
		Vary: 'Origin',
	};
}

function json(data: unknown, status = 200, headers: Record<string, string> = {}) {
	return new Response(JSON.stringify(data), {
		status,
		headers: { 'content-type': 'application/json; charset=utf-8', ...headers },
	});
}

interface AdvisorRequest {
	phase?: string;
	situation?: string;
	hexagramNumber?: number;
	stance?: string;
}

export const OPTIONS: APIRoute = async ({ request }) => {
	return new Response(null, { status: 204, headers: corsHeaders(request) });
};

export const POST: APIRoute = async ({ request }) => {
	const cors = corsHeaders(request);

	let payload: AdvisorRequest;
	try {
		payload = (await request.json()) as AdvisorRequest;
	} catch {
		return json({ error: 'Request body must be valid JSON.' }, 400, cors);
	}

	const situation = typeof payload.situation === 'string' ? payload.situation.trim() : '';
	if (situation.length < 10) {
		return json({ error: 'Please describe your situation in at least a sentence or two.' }, 400, cors);
	}
	if (situation.length > MAX_SITUATION_LENGTH) {
		return json({ error: `Please keep your description under ${MAX_SITUATION_LENGTH} characters.` }, 400, cors);
	}
	if (payload.phase !== 'analyse' && payload.phase !== 'advise') {
		return json({ error: 'Unknown phase. Expected "analyse" or "advise".' }, 400, cors);
	}

	const apiKey = env.GEMINI_API_KEY;
	if (!apiKey) {
		return json({ error: 'The advisor is not configured: the GEMINI_API_KEY secret is missing.' }, 500, cors);
	}

	const ai = new GoogleGenAI({ apiKey });

	try {
		if (payload.phase === 'analyse') {
			return await analyse(ai, situation, cors);
		}
		return await advise(ai, situation, payload.hexagramNumber, payload.stance, cors);
	} catch (error) {
		console.error('Advisor request failed:', error);
		return json({ error: 'The advisor could not reach its reasoning engine. Please try again.' }, 502, cors);
	}
};

/**
 * Step 1 + 2: map the free-text situation to one of the 64 hexagrams and reveal
 * the underlying dynamic. The classical text is then loaded from local data.
 */
async function analyse(ai: GoogleGenAI, situation: string, cors: Record<string, string>) {
	const response = await ai.models.generateContent({
		model: MODEL,
		contents: `A person has described their situation below, in their own words.

Map it to exactly ONE of the 64 hexagrams of the I-Ching (King Wen sequence): choose the hexagram whose underlying dynamic best illuminates the pattern they are living through, not a literal answer to their problem.

Then complete the first two steps of your method:
- situationSummary: their Current Situation, reflected back warmly and specifically.
- underlyingDynamic: the Underlying Dynamic the hexagram reveals, in plain language.

PERSON'S OWN WORDS:
"""
${situation}
"""`,
		config: {
			systemInstruction: SYSTEM_INSTRUCTION,
			maxOutputTokens: 2048,
			responseMimeType: 'application/json',
			responseSchema: responseSchemaPhase1,
		},
	});

	const result = JSON.parse(response.text ?? '{}') as {
		hexagramNumber?: number;
		situationSummary?: string;
		underlyingDynamic?: string;
	};

	const gua = hexagrams.find((h) => h.number === result.hexagramNumber);
	if (!gua) {
		return json({ error: 'The advisor produced an unrecognised mapping. Please try again.' }, 502, cors);
	}

	return json(
		{
			phase: 'analyse',
			situationSummary: result.situationSummary ?? '',
			underlyingDynamic: result.underlyingDynamic ?? '',
			gua: {
				number: gua.number,
				name: gua.name,
				pinyin: gua.pinyin,
				english: gua.english,
				upperTrigram: gua.upperTrigram,
				lowerTrigram: gua.lowerTrigram,
				binary: gua.binary,
				judgment: gua.judgment,
				judgmentEn: gua.judgmentEn,
				image: gua.image,
			},
		},
		200,
		cors,
	);
}

/**
 * Step 3: given the hexagram, the user's chosen posture and the original
 * situation, map to one of the 384 yao (lines) and advise an adaptive strategy.
 */
async function advise(
	ai: GoogleGenAI,
	situation: string,
	hexagramNumber: unknown,
	stance: unknown,
	cors: Record<string, string>,
) {
	if (!isStance(stance)) {
		return json({ error: 'Unknown stance. Expected "positive", "neutral" or "negative".' }, 400, cors);
	}

	const gua = hexagrams.find((h) => h.number === hexagramNumber);
	if (!gua) {
		return json({ error: 'Unknown hexagram. Please analyse your situation first.' }, 400, cors);
	}

	const lines = gua.lines
		.map((line) => `${line.position}. ${line.text} - ${line.textEn}`)
		.join('\n');

	const response = await ai.models.generateContent({
		model: MODEL,
		contents: `A person is working with hexagram ${gua.number}, "${gua.name}" (${gua.english}).

Their situation, in their own words:
"""
${situation}
"""

The six lines (yao) of this hexagram, numbered from the bottom (1) to the top (6):
${lines}

They have decided to meet the underlying dynamic with ${STANCES[stance]}.

Choose the ONE line (1-6) that offers the wisest guidance for someone taking that approach, and complete step 3 of your method:
- lineNumber: the number of that line, counted from the bottom.
- adaptiveStrategy: the Adaptive Strategy - how this person can cope with and adapt to their situation, given that line and the approach they chose. Make it warm, concrete and immediately actionable.`,
		config: {
			systemInstruction: SYSTEM_INSTRUCTION,
			maxOutputTokens: 2048,
			responseMimeType: 'application/json',
			responseSchema: responseSchemaPhase2,
		},
	});

	const result = JSON.parse(response.text ?? '{}') as {
		lineNumber?: number;
		adaptiveStrategy?: string;
	};

	const yao = gua.lines.find((line) => line.position === result.lineNumber);
	if (!yao) {
		return json({ error: 'The advisor produced an unrecognised mapping. Please try again.' }, 502, cors);
	}

	return json(
		{
			phase: 'advise',
			adaptiveStrategy: result.adaptiveStrategy ?? '',
			yao: {
				position: yao.position,
				text: yao.text,
				textEn: yao.textEn,
				guaNumber: gua.number,
				guaName: gua.name,
				guaEnglish: gua.english,
			},
		},
		200,
		cors,
	);
}

function isStance(value: unknown): value is Stance {
	return typeof value === 'string' && value in STANCES;
}
