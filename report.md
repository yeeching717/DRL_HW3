# HW3 理解報告

## HW3-1：Naive DQN（static）+ Experience Replay
- Naive DQN：用單一 Q 網路估計 $Q(s,a)$。每一步用 epsilon-greedy 選動作、前進一步，並用目標 $y = r + \gamma \max_{a'} Q(s',a')$（非終止時）更新當前動作的 Q 值。方法簡單但不穩定，因為目標與更新都依賴同一個網路，且資料高度相關。
- Experience Replay：把轉移 $(s,a,r,s',done)$ 存進緩衝區，再隨機抽小批量訓練。能打破時間相關性、提升樣本效率、改善穩定性。

## HW3-2：強化版 DQN（player mode）
- Double DQN：由線上網路選動作，但用目標網路評估該動作的 Q 值，減少標準 DQN 的高估偏差。
- Dueling DQN：將網路拆成狀態價值 $V(s)$ 與優勢函數 $A(s,a)$，並組合成 $Q(s,a)=V(s)+(A(s,a)-\mathrm{mean}_a A(s,a))$。能更有效地學習狀態價值，尤其在多動作效果接近時有幫助。

## HW3-3：Keras DQN（random mode）與訓練技巧
- 以 Keras 重新實作 DQN，保留目標網路與 Experience Replay。
- 加入穩定化技巧：Huber loss、梯度裁剪（clipnorm）、學習率排程（ExponentialDecay）。
- 訓練邏輯與 PyTorch 版相同；框架差異主要影響程式結構與訓練工具，不改變演算法本身。

## 備註
- Static mode 最簡單，適合用來驗證 DQN 基本行為。
- Player 與 Random mode 初始狀態多變，因此 Replay 與目標網路的重要性更高。

## HW3-4：Rainbow DQN（random mode）（Keras）
- 目標：在 Random Mode GridWorld 上用「簡化版 Rainbow」提升收斂速度與穩定性，沿用既有 Keras 結構。
- 組合內容（不含 Distributional）：Double DQN + Dueling + Prioritized Replay + Multi-step return + NoisyNet。
- 預期改善：
  - Prioritized Replay：用 TD error 作為抽樣優先度，讓學習集中在高誤差經驗上，提升樣本效率。
  - Multi-step return：回饋可往前傳遞更遠，對稀疏回饋與探索更有幫助。
  - NoisyNet：以參數噪音取代 epsilon 探索，動作選擇更穩定且可持續探索。
  - Double + Dueling：降低 Q 值高估，拆解狀態價值與動作優勢以提升學習品質。
- Random Mode 注意點：起始狀態多樣、回饋稀疏，因此 Replay 的品質（PER）與回饋傳播速度（n-step）是主要影響因素。
- 實作規劃（沿用現有結構）：
  - 保留 Keras 訓練主迴圈與 target network。
  - 將 ReplayBuffer 替換為 Prioritized Replay（需維護 priority 與 IS weight）。
  - 加入 n-step buffer，將 $(s,a)$ 對應的 n-step return 與 $s_{t+n}$ 存入 PER。
  - 模型改為 Dueling 結構，並用 Double DQN 的目標計算方式。
  - 使用 NoisyNet 取代 epsilon（或保留小 epsilon 作為保底）。

  ## 實作補充
  - 本專案提供訓練結果圖（Loss / Avg Return / Win Rate）與路徑圖的靜態展示頁。
  - 訓練與輸出可透過 run_generate_outputs.py 一鍵更新圖檔。
  - 目前預設為 4x4、無障礙設定作為基準版本。
