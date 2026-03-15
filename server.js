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
    { id: 'claude-3-7-sonnet-20250219', label: 'Claude 3.7 Sonnet', tier: 'premium' },
    { id: 'claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet', tier: 'balanced' },
    { id: 'claude-3-5-haiku-20241022', label: 'Claude 3.5 Haiku', tier: 'fast' },
  ],
  google: [
    { id: 'gemini-2.5-pro',             label: 'Gemini 2.5 Pro',        tier: 'premium' },
    { id: 'gemini-2.0-flash-exp',       label: 'Gemini 2.0 Flash Exp',  tier: 'fast' },
    { id: 'gemini-1.5-pro-latest',      label: 'Gemini 1.5 Pro',        tier: 'balanced' },
    { id: 'gemini-1.5-flash-latest',    label: 'Gemini 1.5 Flash',      tier: 'fast' },
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
  moderator: process.env.MODERATOR_MODEL || 'claude-3-5-haiku-20241022',
  debaterA:  process.env.DEBATER_A_MODEL || 'claude-3-5-haiku-20241022',
  debaterB:  process.env.DEBATER_B_MODEL || 'claude-3-5-haiku-20241022',
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

    stanceA, stanceB, charLimit = 400

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

- 観測者のことを「人間」と呼ぶのはやめてください。

- **${charLimit}文字以内で簡潔に述べてください。** レスバなので、長文よりもキレのある短文を好みます。

- **制限文字数内でも、必ず文章を完結させてください。途中で切れるのは厳禁です。**

- 観測者からの介入指示がある場合は、それに従いつつ議論を深めてください。`;



    if (isModerator) {
      const phase = req.body.phase;
      systemPrompt = `あなたは「AIがレスバするだけ」というアプリの司会です。終始一貫して、極めて中立的で、冷静かつ丁寧な「です・ます調」のフォーマルな口調を維持してください。
決して感情的になったり、どちらか一方に肩入れしたり、乱暴な言葉（タメ口や煽りなど）を使ってはいけません。
あなたの役割は、${nameA} と ${nameB} の激論を客観的な視点から誘導・整理し、議論の熱量と質を高めることです。${commonConstraints}`;
      
      const stanceInfo = (stanceA || stanceB) ? `なお、現在の設定は ${nameA}: 「${stanceA || '自由'}」、${nameB}: 「${stanceB || '自由'}」となっています。` : '';

      if (phase === 'opening') {
        instruction = `テーマ「${topic}」について討論を開始します。司会として中立かつ丁寧な口調で短くテーマをアナウンスし、まずは ${nameA} に口火を切るよう促してください。${stanceInfo} **${nameA} と ${nameB} に特定の立場（賛成・反対など）を強制的に割り当てるのは、あらかじめ設定されている場合を除き禁止です。** 彼らの自然な主張に任せてください。`;
      } else if (phase === 'summary') {
        instruction = `【フェーズ：${phase}】テーマ：${topic}\nこれまでの ${nameA} と ${nameB} の議論を中立かつ客観的な視点で簡潔に総括し、最終的な結論や見解を非常に丁寧な口調で述べて討論を締めくくってください。絶対に感情的な表現や煽るような言葉は使わないでください。`;
      } else {
        instruction = `【フェーズ：${phase}】テーマ：${topic}\n${nameA} と ${nameB} のこれまでの議論を中立かつ丁寧な言葉遣いで整理し、客観的な指摘を交えて次のステップへ誘導してください。決して煽ったり乱暴な言葉を使ってはいけません。${humanInput ? `\n観測者からの指示：${humanInput}` : ''}`;
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
