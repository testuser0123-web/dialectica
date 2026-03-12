# Dialectica — AI対AI 討論プラットフォーム

## 起動方法（APIキーが手元に来たら）

### 1. APIキーを設定する

```bash
cp .env.example .env
```

`.env` を開いて3つのキーを入力：

```
OPENAI_API_KEY=sk-...        # GPT（モデレーター）
ANTHROPIC_API_KEY=sk-ant-... # Claude（討論者A）
GEMINI_API_KEY=AIza...       # Gemini（討論者B）
```

### 2. 依存パッケージのインストール（初回のみ）

```bash
npm install
```

### 3. サーバーを起動

```bash
node server.js
```

### 4. ブラウザで開く

```
http://localhost:3000
```

---

## 討論の流れ（GPTモデレーター制御）

| フェーズ | 内容 |
|---|---|
| 開幕 | GPTがテーマの論点を整理・提示 |
| 第1ラウンド | Claude・Geminiが初期主張 |
| 第2ラウンド | 相互批判 |
| 転換点 | GPTが新たな問題を提起 |
| 第3ラウンド | 発展的論議 |
| 収束 | GPTが妥協点を誘導 |
| 合意 | 両AIが結論を形成 |
| 終幕 | GPTが総括 |

---

## APIコストの目安

1セッション（8フェーズ）あたり：
- Claude Sonnet：$0.05〜0.15
- Gemini 1.5 Pro：$0.03〜0.10
- GPT-4o-mini：$0.01以下
- **合計：数十円/セッション**

---

## 今後の拡張予定
- [ ] 人間の介入（議題インジェクション）のAPI連携
- [ ] 討論ログの保存・シェア機能
- [ ] モデル選択UI（GPT-4o / Claude Opus / Gemini Ultra など）
- [ ] 収束度の自動終了判定
