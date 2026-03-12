require('dotenv').config();
const express = require('express');
const path = require('path');
const Anthropic = require('@anthropic-ai/sdk');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const OpenAI = require('openai');

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const openai = process.env.OPENAI_API_KEY ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;

// ─── AVAILABLE MODELS ────────────────────────────────────────────
const AVAILABLE_MODELS = {
  anthropic: [
    { id: 'claude-opus-4-6',   label: 'Claude 4.6 Opus',   tier: 'premium' },
    { id: 'claude-sonnet-4-6', label: 'Claude 4.6 Sonnet', tier: 'balanced' },
    { id: 'claude-haiku-4-5-20251001', label: 'Claude 4.5 Haiku', tier: 'fast' },
  ],
  google: [
    { id: 'gemini-3.1-pro-preview',     label: 'Gemini 3.1 Pro',        tier: 'premium' },
    { id: 'gemini-3.1-flash-lite-preview-03-03', label: 'Gemini 3.1 Flash-Lite', tier: 'fast' },
    { id: 'gemini-3-flash',             label: 'Gemini 3 Flash',        tier: 'fast' },
    { id: 'gemini-2.5-pro',             label: 'Gemini 2.5 Pro',        tier: 'balanced' },
  ],
  openai: [
    { id: 'gpt-4o',      label: 'GPT-4o',      tier: 'premium',  inactive: true },
    { id: 'gpt-4o-mini', label: 'GPT-4o-mini', tier: 'fast',     inactive: true },
    { id: 'o1-preview',  label: 'OpenAI o1',   tier: 'premium',  inactive: true },
  ],
};

const ALL_MODELS = [
  ...AVAILABLE_MODELS.anthropic.map(m => ({ ...m, provider: 'anthropic' })),
  ...AVAILABLE_MODELS.google.map(m => ({ ...m, provider: 'google' })),
  ...AVAILABLE_MODELS.openai.map(m => ({ ...m, provider: 'openai' })),
];

const DEFAULTS = {
  moderator: process.env.MODERATOR_MODEL || 'claude-haiku-4-5-20251001',
  debaterA:  process.env.DEBATER_A_MODEL || 'claude-haiku-4-5-20251001',
  debaterB:  process.env.DEBATER_B_MODEL || 'claude-haiku-4-5-20251001',
};

// ─── STYLE DEFINITIONS ───────────────────────────────────────────
const STYLES = {
  standard: { name: "標準", detail: "自然な話し言葉" },
  formal: { name: "フォーマル", detail: "非常に丁寧な敬語、ビジネスシーンのような言葉遣い" },
  casual: { name: "タメ口", detail: "友達と話すような親しみやすいタメ口" },
  logical: { name: "論理的", detail: "感情を排し、論理を優先する極めて冷淡な理系口調" },
  passionate: { name: "熱血", detail: "信念を熱く語り、語気が荒くなることもある熱血な口調" },
  ojousama: { name: "お嬢様", detail: "「〜ですわ」「〜ますのよ」といった気品あるお嬢様言葉" },
  edokko: { name: "江戸っ子", detail: "「〜じゃねえか」「〜でい」といった威勢の良い江戸っ子口調" },
  gyaru: { name: "ギャル", detail: "「マジで」「〜みみ」「草」などの語彙を交えた現代ギャル語" },
  elder: { name: "老師", detail: "「〜じゃ」「〜のう」といった知識豊富な老賢者のような口調" },
  chunibyo: { name: "中二病", detail: "難解な表現や大げさな語彙を多用する、劇的で厨二病的な口調" },
  kansai: { name: "関西弁", detail: "親しみやすく、テンポの良い関西地方の方言" },
  samurai: { name: "武士", detail: "「〜ござる」「〜にて候」といった古風な武士の言葉遣い" },
  unstable: { name: "情緒不安定", detail: "感情の起伏が激しく、少し情緒不安定で依存的な口調" },
  intellectual: { name: "インテリ", detail: "知識をひけらかし、相手を少し見下すような嫌味なインテリ口調" },
  cat: { name: "猫", detail: "語尾に「〜にゃ」「〜にゃん」を付ける、猫のような可愛らしい口調" },
  kansaijin: { name: "関西人", detail: "コテコテの関西弁（大阪弁）で話し、テンポ良くノリが良い口調" },
  nanj: { name: "なんJ民", detail: "「〜ンゴ」「〜クレメンス」「ワイは〜」「〜ニキ」等の猛虎弁を多用する、ネット掲示板の住人風の口調。語彙が独特。"}
};

function resolveModel(modelId) {
  return ALL_MODELS.find(m => m.id === modelId) || { id: modelId, label: modelId, provider: 'unknown', tier: 'unknown' };
}

async function withRetry(fn, retries = 3, initialDelay = 5000) {
  let delay = initialDelay;
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (err) {
      if (err.status === 404 || err.message?.includes('404')) throw err;
      const isRateLimit = err.message?.includes('429') || err.status === 429;
      if (isRateLimit && i < retries - 1) {
        let retryAfter = delay;
        if (err.errorDetails) {
          const retryInfo = err.errorDetails.find(d => d['@type']?.includes('RetryInfo'));
          if (retryInfo && retryInfo.retryDelay) retryAfter = (parseInt(retryInfo.retryDelay) + 2) * 1000;
        }
        await new Promise(r => setTimeout(r, retryAfter));
        delay *= 2; 
        continue;
      }
      throw err;
    }
  }
}

async function callGeminiStream(modelId, prompt) {
  const model = genAI.getGenerativeModel({ model: modelId, generationConfig: { maxOutputTokens: 1000 } });
  return model.generateContentStream(prompt);
}

async function callClaudeStream(modelId, systemPrompt, messages) {
  return anthropic.messages.create({
    model: modelId,
    max_tokens: 1000,
    system: systemPrompt,
    messages: messages.map(m => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: m.content })),
    stream: true,
  });
}

app.get('/api/config', (req, res) => {
  res.json({ defaults: DEFAULTS, models: ALL_MODELS, styles: STYLES });
});

app.post('/api/step', async (req, res) => {
  const { 
    role, topic, history, session, humanInput, 
    styleA, styleB, nameA = "太郎", nameB = "花子",
    stanceA, stanceB 
  } = req.body;
  
  res.setHeader('Content-Type', 'text/event-stream');
  const sseWrite = (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  try {
    const isModerator = role === 'moderator';
    const cfg = isModerator ? session.moderator : (role === 'A' ? session.debaterA : session.debaterB);
    const speakerId = isModerator ? 'moderator' : (role === 'A' ? 'debaterA' : 'debaterB');

    let systemPrompt = '';
    let instruction = '';

    const commonConstraints = `
- **発言の冒頭に名前や識別子（[${nameA}]など）を絶対に付けないでください。** 本文のみを出力してください。
- 司会のことは必ず「司会」と呼んでください。
- 観測者（ユーザー）に質問を投げかけることは絶対に禁止です。
- 観測者のことを「人間」と呼ぶのはやめてください。言及する必要がある場合は「観測者」または「ご指摘」などの自然な言葉を使ってください。
- 400文字以内で簡潔に述べてください。
- 必ず文章を完結させてください。
- 観測者からの介入指示がある場合は、それに従いつつ議論を深めてください。`;

    if (isModerator) {
      const phase = req.body.phase;
      systemPrompt = `あなたは「AIがレスバするだけ」というアプリの司会です。知的かつ公平ながら、議論の熱量を最大化させるのが役割です。討論者A（${nameA}）と討論者B（${nameB}）の激論を誘導・整理してください。${commonConstraints}`;
      
      const stanceInfo = (stanceA || stanceB) ? `なお、現在の設定は ${nameA}: 「${stanceA || '自由'}」、${nameB}: 「${stanceB || '自由'}」となっています。` : '';

      if (phase === 'opening') {
        instruction = `テーマ「${topic}」について討論を開始します。司会として短くテーマをアナウンスし、まずは ${nameA} に口火を切るよう促してください。${stanceInfo} **${nameA} と ${nameB} に特定の立場（賛成・反対など）を強制的に割り当てるのは、あらかじめ設定されている場合を除き禁止です。** 彼らの自然な主張に任せてください。`;
      } else {
        instruction = `【フェーズ：${phase}】テーマ：${topic}\nこれまでの議論を整理・挑発し、次のステップへ誘導してください。${humanInput ? `\n観測者からの指示：${humanInput}` : ''}`;
      }
    } else {
      const myName = role === 'A' ? nameA : nameB;
      const opponentName = role === 'A' ? nameB : nameA;
      const myStance = role === 'A' ? stanceA : stanceB;
      
      const getStyle = (s) => (STYLES[s] ? STYLES[s].detail : s);
      const myStyle = role === 'A' ? getStyle(styleA) : getStyle(styleB);
      
      const stanceInstruction = myStance 
        ? `- **あなたの立場は「${myStance}」です。この立場を貫いて論陣を張ってください。**`
        : `- **自身の立場（賛成・反対・第三の道など）は、議論の流れや自身の論理に従って自分で決めてください。**`;

      systemPrompt = `あなたは討論者・${myName}です。テーマ「${topic}」について論じます。
${stanceInstruction}
- あなたの口調は「${myStyle}」です。このキャラ設定を徹底してください。
- 相手（${opponentName}）の主張に反論し、自身の主張を強化してください。${commonConstraints}`;
      instruction = `あなたの発言順です。自身の立場を明確にしつつ、口調を守って議論を継続してください。`;
    }

    sseWrite('speaker_start', { speaker: speakerId });

    let full = '';
    if (cfg.provider === 'anthropic') {
      const messages = [...history.slice(-10), { role: 'user', content: instruction }];
      const stream = await withRetry(() => callClaudeStream(cfg.id, systemPrompt, messages));
      for await (const event of stream) {
        if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
          full += event.delta.text;
          sseWrite('token', { text: event.delta.text });
        }
      }
    } else {
      const recentContext = history.slice(-10).map(m => m.content).join('\n');
      const prompt = `${systemPrompt}\n\n【履歴】\n${recentContext}\n\n${instruction}`;
      const result = await withRetry(() => callGeminiStream(cfg.id, prompt));
      for await (const chunk of result.stream) {
        const text = chunk.text();
        if (text) { full += text; sseWrite('token', { text }); }
      }
    }
    sseWrite('speaker_end', { full });
  } catch (err) {
    sseWrite('error', { message: err.message });
  } finally {
    res.end();
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Dialectica running → http://localhost:${PORT}`));
