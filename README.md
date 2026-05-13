# DRL HW3 專案說明

## 專案網頁連結
- https://yeeching717.github.io/DRL_HW3/

## 專案簡介
本專案包含 HW3-1 ~ HW3-4 的 DQN 實作與視覺化展示，並提供訓練結果圖與 GridWorld 路徑圖的靜態網頁。

## 執行環境
- Python 3.12
- 套件：numpy、matplotlib、torch、tensorflow

## 檔案結構
- hw3_1.py：Naive DQN + Experience Replay
- hw3_2.py：Double DQN / Dueling DQN
- hw3_3.py：Keras DQN
- hw3_4.py：Rainbow DQN（簡化版）
- run_generate_outputs.py：產生訓練圖與勝率、輸出到 site/assets
- report.md：HW3 理解報告
- 聊天紀錄.md：對話紀錄
- site/：本機靜態網站
- docs/：GitHub Pages 使用的靜態網站

## 快速使用
1) 產生訓練結果圖與勝率
```bash
python run_generate_outputs.py
```
輸出圖片會更新在 site/assets。

2) 本機預覽網頁
- 開啟 site/index.html

3) GitHub Pages
- docs/ 目錄提供 Pages 版本（已與 site/ 同步）

## 注意事項
- 目前設定為 4x4、無障礙版本。
- 訓練輸出圖檔：
  - site/assets/hw3_1_train.png
  - site/assets/hw3_2_train.png
  - site/assets/hw3_3_train.png
  - site/assets/hw3_4_train.png
