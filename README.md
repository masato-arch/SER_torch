# このリポジトリについて
大学院で研究してる音声感情認識のプロジェクトを管理するためのリポジトリです．コードはmasterに入れていきます．

# 研究背景と目的
人間の発話音声には様々な情報が含まれる.1) 発話 内容，2) 話者が誰であるか，3) 感情的な情報，4) 発 話の流暢さや明瞭さなど，多種多様な情報を含んでいる.来る Human Computer Interaction (HCI)の時代において，機械がこれらの情報を総合的に活用して人間とコミュニケーションをとることは大変重要である. AlexaやSiri，YouTubeの自動字幕システムなどをはじめとする今日の音声認識システムは，発話内容を正確に認識することには大変優れている.次いでAlexaやSiriが持ち主の声のみに反応するように，話者が誰であるかを聞き分けることも得意である.音声感情認識においても近年はニューラルネットワークを用いた研究が盛んであり，年々認識率の向上が見られるが，まだ十分な認識率を得られていない. そこで本研究ではデータセットや特徴量，モデル構造を包括的に見直し，精度の向上を図る．  

# 現在の取り組み

Tensor Factorized Neural Network (TFNN)を用いた軽量かつ高精度なモデル
> S. K. Pandey, H. S. Shekhawat and S. Shekhawat, "Attention gated tensor neural network architectures for speech emotion recognition," Biomedical Signal Processing and Control, Volume 71, Part A, 2022, 103173, ISSN 1746-8094, https://doi.org/10.1016/j.bspc.2021.103173.

の実装と検証

