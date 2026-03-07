# 🤖 Bitkub Crypto Auto Trading Bot

ระบบ Auto Trading Bot สำหรับ Bitkub Exchange พร้อม AI (LSTM + Reinforcement Learning) สำหรับพยากรณ์ราคาและตัดสินใจซื้อขายอัตโนมัติ พร้อมหน้า GUI แบบ Dark Theme

---

## 📁 Project Structure

```
bitkub-trading-bot/
├── main.py                     # Entry point (trade / gui / backtest / train)
├── gui.py                      # GUI Dashboard (Tkinter dark theme)
├── config.py                   # Configuration (API, Trading, Risk, AI)
├── requirements.txt            # Dependencies
├── .env.example                # Environment variables template
│
├── exchange/                   # Exchange API Module
│   ├── bitkub_client.py        # Bitkub REST API client (direct, ไม่ใช้ ccxt)
│   └── data_collector.py       # Market data collector
│
├── strategy/                   # Trading Strategy Module
│   ├── indicators.py           # Technical Indicator Engine (RSI, MACD, BB, EMA)
│   ├── trading_strategy.py     # Buy/Sell/SL/TP logic + Position tracking
│   └── risk_management.py      # Dynamic Risk Management
│
├── ai_model/                   # AI Models
│   ├── lstm_model.py           # LSTM price prediction
│   ├── rl_model.py             # RL trading decision agent (DQN)
│   └── saved/                  # Saved model weights
│
├── backtest/                   # Backtesting System
│   └── backtester.py           # Strategy backtester
│
├── dashboard/                  # Trading Dashboard
│   └── trading_dashboard.py    # Console dashboard
│
├── utils/                      # Utilities
│   └── logger.py               # Logging system
│
├── data/                       # Market data storage
└── logs/                       # Log files
```

---

## 🚀 วิธีติดตั้ง

### 1. ติดตั้ง Python 3.11+

ดาวน์โหลดจาก [python.org](https://www.python.org/downloads/)

### 2. สร้าง Virtual Environment

```powershell
cd bitkub-trading-bot
python -m venv venv
.\venv\Scripts\Activate
```

### 3. ติดตั้ง Dependencies

```powershell
pip install -r requirements.txt
```

### 4. ติดตั้ง PyTorch (เลือกตาม GPU)

**CPU only:**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU (CUDA 12.1):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 🔑 วิธีใส่ Bitkub API

### 1. สร้าง API Key จาก Bitkub

1. ไปที่ [Bitkub.com](https://www.bitkub.com) → Settings → API Keys
2. สร้าง API Key ใหม่
3. เปิดสิทธิ์ **Read** และ **Trade**
4. **⚠️ อย่าเปิดสิทธิ์ Withdraw** (เพื่อความปลอดภัย)

### 2. ใส่ API Key

มี 3 วิธี:

**วิธีที่ 1: ใส่ผ่าน GUI (แนะนำ)**
- เปิดโปรแกรมด้วย `python main.py gui`
.venv\Scripts\python.exe main.py gui
- ใส่ API Key และ Secret ในช่อง 🔑 API SETTINGS
- รองรับ **Ctrl+V วาง** และ **คลิกขวา > Paste**
- กดปุ่ม 👁 เพื่อดู/ซ่อนรหัสได้

**วิธีที่ 2: ใช้ .env file**
```powershell
copy .env.example .env
# แก้ไขไฟล์ .env ใส่ API Key และ Secret
```

**วิธีที่ 3: ตั้งค่าผ่าน PowerShell**
```powershell
$env:BITKUB_API_KEY = "your_api_key_here"
$env:BITKUB_API_SECRET = "your_api_secret_here"
```

---

## 🏃 วิธีรัน Bot

### 🖥️ รัน GUI Dashboard (แนะนำ)

```powershell
python main.py gui
```

### รัน Auto Trading (Console)

```powershell
python main.py trade
```

### รัน Auto Trading พร้อมตั้งค่า

```powershell
python main.py trade --symbol BTC_THB --interval 30 --sl 3.0 --tp 5.0
```

### รัน Backtest

```powershell
python main.py backtest
```

### Train AI Model

```powershell
python main.py train
```

---

## 🖥️ คู่มือใช้งาน GUI Dashboard

### เปิดโปรแกรม

```powershell
python main.py gui
```

จะเปิดหน้าต่าง GUI แบบ Dark Theme ขนาด 1280x850 pixel

### แผงควบคุมหลัก

GUI แบ่งเป็น **2 ฝั่ง**:

| ฝั่งซ้าย | ฝั่งขวา |
|-----------|----------|
| 📊 Market Data (ราคา, Balance, P/L) | 🔑 API Settings |
| 💰 Wallet Balance (ตารางเหรียญ) | 💸 Quick Trade (ซื้อ/ขายเอง) |
| 📈 Indicators (RSI, MACD, BB, EMA) | 🧠 AI Prediction |
| 📋 Positions (ตำแหน่งเปิด) | ⚙️ Controls (ตั้งค่า Bot) |
| 💹 Real-Time P/L (กำไร/ขาดทุน) | ⚠️ Risk Status |
| 📝 Activity Log | 📜 Trade History |

---

### ขั้นตอนที่ 1: เชื่อมต่อ Exchange

1. ใส่ **API Key** ในช่อง (Ctrl+V หรือคลิกขวา > Paste)
2. ใส่ **API Secret** ในช่อง
3. กดปุ่ม 👁 ตรวจสอบว่าใส่ถูกต้อง
4. เลือก **เหรียญที่จะเทรด** (เช่น BTC_THB, ETH_THB)
5. กดปุ่ม **🔗 Connect & Load Wallet**
6. ถ้าสำเร็จจะแสดง ✅ Connected และโชว์ยอดเงินใน Wallet

---

### ขั้นตอนที่ 2: ซื้อเหรียญ (Quick Trade)

หลังเชื่อมต่อแล้ว จะเปิดใช้งาน **💸 QUICK TRADE**:

1. ใส่จำนวนเงิน THB ที่ต้องการซื้อ (ขั้นต่ำ 10 THB)
2. กด **📈 BUY NOW** → ยืนยัน → Bot ซื้อเหรียญทันที
3. หลังซื้อ Bot จะ **เริ่มทำงานอัตโนมัติ** เพื่อเฝ้าดูราคา
4. Position จะถูกลงทะเบียนพร้อม SL/TP ตามค่าที่ตั้ง

**ขายเหรียญ:**
- กด **📉 SELL ALL** → ขายเหรียญทั้งหมดของเหรียญที่เลือก

---

### ขั้นตอนที่ 3: ตั้งค่า Bot

ใน **⚙️ CONTROLS**:

| ค่า | คำอธิบาย | ค่าเริ่มต้น |
|------|----------|-------------|
| Interval (s) | ความถี่ตรวจสอบราคา (วินาที) | 30 |
| SL % | Stop Loss สำรอง ถ้า AI ยังไม่สั่งขาย | 1.8 |
| TP % | Take Profit — กำไรกี่ % จะขายทำกำไร | 5.0 |
| Auto Buy (THB) | จำนวนเงินที่ Auto Bot ใช้ซื้อเมื่อ AI ให้สัญญาณ BUY | 100 |

หมายเหตุ:
- ระบบปิดสถานะด้วยการส่งคำสั่งขายในคู่ THB เช่น BTC_THB เพื่อแปลงเหรียญกลับเป็นเงินบาทใน Bitkub
- จึงไม่ใช่การถอนเหรียญออกนอกระบบ และไม่เกี่ยวกับค่าธรรมเนียมถอน
- เมื่อ Bot กำลังทำงานอยู่ การแก้ค่าในพารามิเตอร์จะมีผลทันทีในรอบถัดไป โดยไม่ต้อง Stop/Start ใหม่
- ค่า `Interval` ใหม่จะถูกนำไปใช้ระหว่างรอรอบทันที ส่วนการเปลี่ยน `Symbol` จะอนุญาตเฉพาะตอนที่ไม่มี position ค้างอยู่
- Badge ข้างพารามิเตอร์: `EDIT` = มีการแก้แต่ยังไม่รัน, `PENDING` = เปลี่ยนแล้วกำลังรอใช้, `LIVE` = ค่าในช่องถูกใช้จริงแล้ว, `INVALID` = ค่าที่กรอกไม่ถูกต้อง

---

### ขั้นตอนที่ 4: เปิด Boss Mode 🏆

**Boss Mode** คือระบบเทรดอัตโนมัติแบบ AI CutLoss + Re-Buy วนรอบ

| ค่า | คำอธิบาย | ค่าเริ่มต้น |
|------|----------|-------------|
| 🏆 Boss Mode | เปิด/ปิด Boss Mode | ✅ ON |
| CutLoss % | ขาดทุนกี่ % แล้วให้ AI เริ่มพิจารณาขาย | 0.6 |
| Recovery % | ราคาฟื้นกี่ % จากจุดขาย แล้วค่อยซื้อคืน | 0.8 |

**วิธีทำงานของ Boss Mode:**

```
┌─────────────────────────────────────────────┐
│  ถือเหรียญอยู่                                │
│  ↓                                           │
│  ราคาลง ≥ CutLoss% (เช่น -0.5%)             │
│  → AI ประเมินแนวโน้มลงต่อ + SELL            │
│  → 🔴 ขายกลับเป็น THB ใน Bitkub              │
│  → Bot ยังทำงานต่อ รอราคาฟื้นเพื่อซื้อคืน       │
│  ↓                                           │
│  ราคาฟื้น ≥ Recovery% จากจุดขาย (เช่น +0.5%)│
│  → 🟢 ซื้อคืนอัตโนมัติ (BOSS BUY-BACK)      │
│  → กลับไปถือเหรียญ วนรอบใหม่                  │
└─────────────────────────────────────────────┘
```

**ตัวอย่าง:**
1. ซื้อ BTC ที่ราคา 2,000,000 THB
2. ราคาลงเหลือ 1,990,000 (-0.5%) → AI ยืนยัน cut loss แล้ว Bot ขายกลับเป็น THB อัตโนมัติ
3. Bot รอดูราคา...
4. ราคาขึ้นเป็น 2,000,000 (+0.5% จากจุดขาย) → Bot ซื้อคืน
5. วนรอบต่อไปเรื่อยๆ

> 💡 **เคล็ดลับ:** ปิด Boss Mode ได้ตลอดเวลา → กลับไปใช้ SL/TP แบบปกติ

---

### ขั้นตอนที่ 5: เริ่มใช้ Bot

1. กดปุ่ม **▶ START BOT**
2. Bot จะเริ่มทำงานตามรอบ (ทุก 30 วินาที หรือตามที่ตั้ง)
3. ถ้า AI ประเมินว่าถึงจังหวะ BUY ระบบจะส่งคำสั่งซื้อทันทีด้วยจำนวนเงินจากช่อง **Auto Buy (THB)**
4. ดูข้อมูล real-time ได้ที่:
   - **💹 Real-Time P/L** — กำไร/ขาดทุนปัจจุบัน (สีเขียว = กำไร, สีแดง = ขาดทุน)
   - **📊 Market Data** — ราคาปัจจุบัน, Balance, Total Value
   - **📝 Activity Log** — บันทึกทุกการทำงานของ Bot
   - **📜 Trade History** — ประวัติการซื้อขาย
5. กด **⏹ STOP BOT** เพื่อหยุด Bot

---

### ปุ่มเสริม

| ปุ่ม | คำอธิบาย |
|------|----------|
| 🧠 Train AI | Train โมเดล AI (LSTM + RL) จากข้อมูลจริง |
| 📊 Backtest | ทดสอบกลยุทธ์ย้อนหลัง |
| 🔄 Refresh | รีเฟรชข้อมูลราคาและ Indicators |

---

## 🧠 ระบบ AI

### LSTM Price Prediction

- ใช้ Long Short-Term Memory (LSTM) neural network
- พยากรณ์ทิศทางราคา (ขึ้น/ลง) พร้อม confidence
- Input: ราคา OHLCV + Technical Indicators 60 แท่งย้อนหลัง

### RL Trading Agent (DQN)

- ใช้ Deep Q-Network เรียนรู้กลยุทธ์การเทรด
- ตัดสินใจ BUY / SELL / HOLD โดยอัตโนมัติ
- เรียนรู้จากข้อมูลจริงและปรับตัวตามตลาด

### วิธี Train AI

```powershell
python main.py train
```

หรือกดปุ่ม **🧠 Train AI** ใน GUI

Bot จะ:
1. ดึงข้อมูลราคาย้อนหลัง 1000 แท่ง (1h timeframe)
2. คำนวณ Technical Indicators
3. Train LSTM model สำหรับพยากรณ์ราคา
4. Train RL agent สำหรับตัดสินใจซื้อขาย
5. บันทึก model ที่ `ai_model/saved/`

### ปรับแต่ง AI

แก้ไขใน `config.py` → `AIConfig`:

```python
@dataclass
class AIConfig:
    lstm_sequence_length: int = 60    # จำนวน time steps ย้อนหลัง
    lstm_hidden_size: int = 128       # LSTM hidden units
    lstm_num_layers: int = 2          # จำนวน LSTM layers
    lstm_epochs: int = 50             # จำนวน training epochs
    lstm_batch_size: int = 32         # Batch size
    lstm_learning_rate: float = 0.001 # Learning rate

    rl_episodes: int = 1000           # RL training episodes
    rl_learning_rate: float = 0.0003  # RL learning rate
    rl_gamma: float = 0.99            # Discount factor
```

---

## ⚙️ ระบบการทำงาน

### Auto Trading Loop (ทุก 30 วินาที)

```
1. ดึงราคาจาก Bitkub API
2. คำนวณ Technical Indicators (RSI, MACD, BB, EMA)
3. AI วิเคราะห์ (LSTM prediction)
4. [Boss Mode ON] → ตรวจ CutLoss / Re-Buy
   [Boss Mode OFF] → ตรวจ Stop Loss / Take Profit
5. ตรวจสอบสัญญาณ BUY / SELL signals
6. ส่งคำสั่งซื้อ/ขายผ่าน API (ถ้ามี signal)
7. อัพเดท Real-Time P/L + GUI
```

### เงื่อนไข BUY (Auto)

| Condition | Threshold |
|-----------|-----------|
| RSI | < 35 (oversold) |
| Price vs EMA | ราคาต่ำกว่า EMA 21 |
| AI Prediction | ทิศทางขึ้น + confidence > 30% |
| MACD (bonus) | Bullish crossover |
| BB (bonus) | ราคาต่ำกว่า Bollinger lower band |

*ต้องผ่าน 2 ใน 3 เงื่อนไขหลักถึงจะ BUY*

### เงื่อนไข SELL (Auto)

| Condition | Threshold |
|-----------|-----------|
| RSI | > 70 (overbought) |
| Price vs BB | ราคาสูงกว่า Bollinger upper band |
| AI Prediction | ทิศทางลง + confidence > 30% |
| MACD (bonus) | Bearish crossover |

*ต้องผ่าน 2 ใน 3 เงื่อนไขหลักถึงจะ SELL*

### โหมดการทำงาน

| โหมด | CutLoss | Re-Buy | Take Profit |
|------|---------|--------|-------------|
| **Boss Mode ON** | -0.5% → ขาย, รอซื้อคืน +0.5% | อัตโนมัติ | ตาม TP% |
| **Boss Mode OFF** | ตาม SL% (default -3%) | ไม่มี | ตาม TP% (default +5%) |

### Risk Management

| Parameter | Default |
|-----------|---------|
| Max Trade Size | 10,000 THB |
| Max Daily Loss | 5,000 THB |
| Max Position % | 30% ของ balance |
| Max Open Positions | 3 |

---

## 📊 Backtesting

```powershell
python main.py backtest
```

หรือกดปุ่ม **📊 Backtest** ใน GUI

ผลลัพธ์จะแสดง:
- **Win Rate** — อัตราชนะ
- **Total Return** — ผลตอบแทนรวม
- **Max Drawdown** — การลดลงสูงสุด
- **Sharpe Ratio** — อัตราส่วนผลตอบแทนต่อความเสี่ยง
- **Profit Factor** — อัตราส่วนกำไรต่อขาดทุน

---

## 💹 Real-Time P/L Display

เมื่อ Bot ทำงาน จะแสดงข้อมูล real-time:

| ข้อมูล | คำอธิบาย |
|--------|----------|
| Unrealized P/L (THB) | กำไร/ขาดทุนที่ยังไม่ได้ขาย (บาท) |
| P/L % | เปอร์เซ็นต์กำไร/ขาดทุน |
| Position Summary | รายละเอียด: เหรียญ, จำนวน, ราคาซื้อ → ราคาปัจจุบัน |

- 🟢 **สีเขียว** = กำไร
- 🔴 **สีแดง** = ขาดทุน

อัพเดททุกรอบ (ตาม Interval ที่ตั้ง)

---

## 📝 Logs

Log files จะอยู่ในโฟลเดอร์ `logs/`:

| File | Description |
|------|-------------|
| `trades.log` | บันทึกการซื้อขายทั้งหมด |
| `errors.log` | บันทึก errors |
| `ai_predictions.log` | บันทึก AI predictions |

---

## 🔒 ความปลอดภัย

- API Key ไม่ถูกเก็บในโค้ด — ใส่ผ่าน GUI หรือ .env
- ช่อง API Key/Secret แสดงเป็น •••• (กดปุ่ม 👁 เพื่อดู)
- รองรับ Ctrl+V, คลิกขวา Paste, Ctrl+C/X/A
- ใช้ HMAC-SHA256 (v3) สำหรับ sign API request
- **ห้ามเปิดสิทธิ์ Withdraw ใน API Key**

---

## ⚠️ คำเตือน

- **Bot นี้ใช้เงินจริงในการเทรด** — ทดสอบด้วย backtest ก่อนเสมอ
- **ไม่มีการรับประกันกำไร** — Crypto มีความผันผวนสูง
- **ตั้ง Stop Loss เสมอ** — เพื่อจำกัดความเสียหาย
- **อย่าใช้เงินที่ไม่พร้อมจะเสีย**
- **ตรวจสอบ API Key permissions** — เปิดเฉพาะ Read + Trade, ไม่เปิด Withdraw
- **Boss Mode เหมาะสำหรับตลาด Sideway** — ตลาด Downtrend อาจ cutloss ซ้ำหลายรอบ

---

## 📜 License

MIT License