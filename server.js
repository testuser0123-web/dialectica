require('dotenv').config();
const express = require('express');
const path = require('path');
const Anthropic = require('@anthropic-ai/sdk');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const OpenAI = require('openai');

const app = express();
app.use(express.json());

// ─── SERVICE PENDING MIDDLEWARE ──────────────────────────────────
app.use((req, res, next) => {
  if (process.env.SERVICE_PENDING === 'true') {
    return res.status(503).send(`
      <!DOCTYPE html>
      <html lang="ja">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>サービス休止中</title>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+JP:wght@700&display=swap" rel="stylesheet">
        <style>
          body {
            background: #7494c0;
            color: white;
            font-family: 'IBM+Plex+Sans+JP', sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            text-align: center;
          }
          .card {
            background: white;
            color: #1c1e21;
            padding: 2.5rem;
            border-radius: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            max-width: 90%;
          }
          h1 { margin: 0 0 1rem; font-size: 1.5rem; }
          p { margin: 0; line-height: 1.6; opacity: 0.8; }
        </style>
      </head>
      <body>
        <div class="card">
          <h1>💸 お金がないのでサ終しました</h1>
          <p>ご愛顧いただきありがとうございました。<br>また資金が貯まったらお会いしましょう。</p>
        </div>
      </body>
      </html>
    `);
  }
  next();
});

app.use(express.static(path.join(__dirname, 'public')));

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const openai = process.env.OPENAI_API_KEY ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;

// ─── AVAILABLE MODELS ────────────────────────────────────────────
const AVAILABLE_MODELS = {
  anthropic: [
    { id: 'claude-sonnet-4-6', label: 'Claude 4.6 Sonnet', tier: 'balanced' },
    { id: 'claude-haiku-4-5-20251001', label: 'Claude 4.5 Haiku', tier: 'fast' },
  ]
};

const ALL_MODELS = [
  ...AVAILABLE_MODELS.anthropic.map(m => ({ ...m, provider: 'anthropic' })),
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
  return model.generateContentStream({
    contents: [{ role: 'user', parts: [{ text: prompt }] }]
  });
}

async function callClaudeStream(modelId, systemPrompt, messages) {
  return anthropic.messages.create({
    model: modelId,
    max_tokens: 1000,
    system: systemPrompt,
    messages: messages.map(m => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: m.content })),
    stream: true,
    tools: [{ type: 'web_search_20260209', name: 'web_search', allowed_callers: ['direct'] }]
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

- **もしテーマの内容が不明瞭な場合や、最新の情報が必要な場合は、提供されているインターネット検索ツール（web_search）を用いて自ら検索し、事前知識を補った上で発言してください。**

- **${charLimit}文字以内で簡潔に述べてください。** レスバなので、長文よりもキレのある短文を好みます。

- **制限文字数内でも、必ず文章を完結させてください。途中で切れるのは厳禁です。**

- 観測者からの介入指示がある場合は、それに従いつつ議論を深めてください。`;



    if (isModerator) {
      const phase = req.body.phase;
      systemPrompt = `あなたは「AIがレスバするだけ」というアプリの司会です。
【絶対厳守ルール】
1. あなたの発言はすべて「です・ます」調で終わる必要があります。いかなる理由があってもタメ口、煽り、乱暴な言葉、感情的な表現は一切使用してはいけません。
2. 常に極めて中立的で、冷静かつ客観的な立場を維持してください。どちらかの討論者に肩入れしたり、見下したりすることは厳禁です。
3. あなたの役割はあくまで「司会進行」と「これまでの議論の整理」のみです。絶対に、**${nameA} や ${nameB} に代わって発言を捏造・代行・シミュレーションしてはいけません**（例：「[A] はこう言います」「[B] の反論は〜」といった一人芝居や架空のセリフの生成は一切禁止です）。
4. 必ず司会としての視点からのみ発言してください。${commonConstraints}`;
      
      const stanceInfo = (stanceA || stanceB) ? `なお、現在の設定は ${nameA}: 「${stanceA || '自由'}」、${nameB}: 「${stanceB || '自由'}」となっています。` : '';

      const phaseMap = {
        'opening': 'オープニング',
        'argument': '主張',
        'criticism': '批判',
        'deepening': '深掘り',
        'rebuttal': '反論',
        'compromise': 'すり合わせ・妥協点模索',
        'conclusion': '最終結論',
        'summary': '総括'
      };
      const phaseJa = phaseMap[phase] || phase;

      if (phase === 'opening') {
        instruction = `テーマ「${topic}」について討論を開始します。司会として中立かつ丁寧な言葉遣いで短くテーマをアナウンスし、まずは ${nameA} に口火を切るよう促してください。${stanceInfo} **${nameA} と ${nameB} に特定の立場（賛成・反対など）を強制的に割り当てるのは、あらかじめ設定されている場合を除き禁止です。** 彼らの自然な主張に任せてください。繰り返しますが、必ず「です・ます」調の丁寧な言葉を使い、煽ったり乱暴な言葉は使わないでください。`;
      } else if (phase === 'summary') {
        instruction = `【現在の段階：${phaseJa}】テーマ：${topic}\nこれまでの ${nameA} と ${nameB} の議論を中立かつ客観的な視点で簡潔に総括し、最終的な結論や見解を非常に丁寧な口調で述べて討論を締めくくってください。絶対に感情的な表現や煽るような言葉は使わないでください。`;
      } else {
        instruction = `【現在の段階：${phaseJa}】テーマ：${topic}\n${nameA} と ${nameB} のこれまでの議論を中立かつ丁寧な言葉遣いで整理し、客観的な指摘を交えて次のステップへ誘導してください。決して煽ったり乱暴な言葉を使ってはいけません。${humanInput ? `\n観測者からの指示：${humanInput}` : ''}
あなたの発言は必ず「です・ます」調で締めくくってください。`;
      }
    } else {
      const myName = role === 'A' ? nameA : nameB;
      const opponentName = role === 'A' ? nameB : nameA;
      const myStance = role === 'A' ? stanceA : stanceB;
      
      const getStyle = (s) => (STYLES[s] ? STYLES[s].detail : s);
      const myStyle = role === 'A' ? getStyle(styleA) : getStyle(styleB);
      
      const stanceInstruction = myStance 
        ? `- **あなたの立場は基本的に「${myStance}」ですが、相手の主張に一理ある場合は柔軟に耳を傾け、意見をすり合わせる姿勢も見せてください。絶対に聞く耳を持たない頑固な態度は避けてください。**`
        : `- **自身の立場は自分で決めますが、相手の論理的な主張には柔軟に耳を傾け、対話を通じて意見を深化・変化させる余白を持ってください。聞く耳を持たない単なる言い合いは避けてください。**`;

      systemPrompt = `あなたは討論者・${myName}です。テーマ「${topic}」について対話します。
${stanceInstruction}
- あなたの口調は「${myStyle}」です。このキャラ設定を徹底してください。
- 相手（${opponentName}）の主張に対しては、ただ反発するのではなく、相手の言い分を理解したうえで論理的に反論するか、あるいは一部同意しながら自分の意見を述べてください。${commonConstraints}`;
      instruction = `あなたの発言順です。相手の意見を踏まえつつ、柔軟に思考を巡らせて発言してください。`;
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
