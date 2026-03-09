# Bitkub Auto Trading Bot

บอทเทรด Bitkub ที่รวม GUI, paper trading, AI prediction, fee-aware P/L และระบบป้องกันความเสี่ยงสำหรับตลาดขาลง โดยเวอร์ชันปัจจุบันเน้น 3 เรื่องหลัก:

- ลดการซื้อสวน downtrend ที่ยังไม่ฟื้นจริง
- ลดขนาดไม้โดยอัตโนมัติเมื่อ ATR สูง, เทรนด์ลง, หรือแพ้ติดกัน
- ลดการ cut loss ถี่จากการเข้าไม้เร็วเกินไปและการ buy-back เร็วเกินไป
- กันการซื้อในจังหวะที่ราคาโดนลากลงแรงด้วย dump protection
- ให้ AI ประเมินโซนเข้าซื้อและเปิด BUY ได้ตั้งแต่ช่วงตลาดเริ่มฟื้น ไม่ต้องรอให้ขาขึ้นชัดเจนเกินไป
- เมื่อกำไรถึง threshold ระบบสามารถถอนเฉพาะกำไรกลับเป็น THB และคงเงินต้นไว้ในเหรียญได้

## โครงสร้างโปรเจกต์

```text
Bitkub/
├── main.py
├── gui.py
├── config.py
├── requirements.txt
├── exchange/
├── strategy/
├── ai_model/
├── backtest/
├── dashboard/
├── utils/
├── data/
└── logs/
```

## ความสามารถหลัก

- Technical indicators: RSI, MACD, Bollinger Bands, EMA, support/resistance, ATR, volume ratio
- AI stack: LSTM prediction, RL agent, optional LLM advisor
- Fee-aware position tracking: คำนวณต้นทุนเข้าและมูลค่าออกสุทธิหลังหักค่าธรรมเนียม
- Paper trading: ใช้พอร์ตจำลองแยกจาก wallet จริง
- Boss Mode: AI cut loss ไป THB แล้วรอ buy-back เมื่อราคาฟื้น
- Adaptive risk sizing: ลด position size อัตโนมัติตาม market regime
- GUI runtime controls: เปลี่ยนค่าระหว่างรันได้และระบบจะ apply ให้ทันทีโดยไม่ต้องหยุด auto trade พร้อม badge `EDIT / PENDING / LIVE / INVALID`
- Profit Cashout: ถอนกำไรบางส่วนกลับเป็น THB แล้ว rebase position ให้เงินต้นยังอยู่ในเหรียญเพื่อรอรอบกำไรถัดไป

## สิ่งที่ปรับปรุงในเวอร์ชันนี้

### 0. UI ใช้งานง่ายขึ้น

หน้าจอถูกจัดใหม่ให้ flow ชัดขึ้นกว่าเดิม:

- ฝั่งขวามีการ์ด `Quick Start` บอกสถานะปัจจุบันและขั้นตอนถัดไป
- ฝั่งซ้ายเลื่อนลงได้ ทำให้ดู `Market Data`, `P/L`, `Wallet`, `Indicators`, `Positions`, `Log` ได้ครบแม้จอไม่สูงมาก
- sidebar ฝั่งขวาเลื่อนลงได้ ทำให้การตั้งค่าไม่ล้นหน้าจอในจอเล็ก
- scrollbar ของทั้งสองฝั่งถูกทำให้มองเห็นและลากง่ายขึ้นกว่าก่อน
- หมุนล้อเมาส์ได้จากทุก widget ภายในคอลัมน์ ไม่ต้องเล็งเฉพาะพื้นที่ว่างของ panel
- กด `Shift` + ล้อเมาส์ เพื่อเลื่อนแบบเร็วเป็นหน้าในคอลัมน์ซ้ายหรือขวา
- มีปุ่ม `TOP` และ `BOTTOM` อยู่เหนือคอลัมน์ซ้ายและขวา เพื่อกระโดดไปต้น/ท้าย panel ได้ทันที
- การ์ด `Controls` แบ่งเป็น `Core Settings`, `Automation`, `Recovery`
- การ์ด `Risk Status` อธิบายได้ทันทีว่าระบบกำลังลดไม้หรือพักซื้อเพราะอะไร
- ช่อง `Decision` แสดงเหตุผลซื้อ/รอ/ถือเป็นข้อความไทยสั้น ๆ แยกหลายบรรทัด อ่านง่ายกว่าการต่อเป็นประโยคยาวบรรทัดเดียว
- `Real-Time P/L` ถูกย้ายขึ้นมาอยู่ใกล้ข้อมูลหลักเพื่อดูพอร์ตง่ายขึ้น
- `Quick Trade` มี 2 โหมด: `Manual Buy + Auto Trade` และ `Auto Trade Only`
- ฝั่ง `Automation` มีสวิตช์ `Profit Cashout` และ `Small Fee Guard` ให้เปิด/ปิด พร้อมตั้ง threshold ของไม้เล็กที่ต้องการป้องกันการขายเร็วเกินไปได้ระหว่างบอทรันอยู่
- การ์ดฝั่งขวาแต่ละใบสามารถกดปุ่ม `- / +` ที่หัวการ์ดเพื่อยุบ/ขยายได้ ช่วยประหยัดพื้นที่บนจอเล็ก
- การ์ดสถานะบอทแยก `กำไร/ขาดทุนที่ปิดแล้ว`, `กำไร/ขาดทุนลอยตัว`, และ `กำไร/ขาดทุนรวม` ออกจากกันชัดเจน
- ใต้กล่องสถิติจะบอก `ขาดทุนที่ปิดล่าสุด` และ `ลอยตัวตอนนี้` ว่ามาจากเหรียญไหน ราคาเข้า/ออกเท่าไร เพื่อดูต้นเหตุของ P/L ได้ง่ายขึ้น

### 1. Downtrend Guard

บอทจะไม่รีบสะสมในตลาดลงเหมือนเดิมอีกแล้ว โดยการซื้อสวนเทรนด์จะเกิดได้ต่อเมื่อมีสัญญาณฟื้นตัวชัดเจนพร้อมกัน เช่น:

- ราคา reclaim กลับเหนือ EMA9
- MACD กลับเป็น bullish
- volume ratio แข็งแรง
- AI มองขึ้นด้วยความมั่นใจสูงพอ

ผลคือ bot จะเข้าไม้ช้าลงใน downtrend แต่ลดการโดน cut loss ซ้ำในจุดเด้งหลอกได้ชัดเจนกว่าเดิม

### 1.2 AI Entry Zone และ Early Recovery Entry

เวอร์ชันนี้ AI จะไม่ได้ดูแค่ว่า `ซื้อ` หรือ `ยังไม่ซื้อ` แต่จะช่วยประเมินด้วยว่า:

- ควรรอเข้าซื้อแถวไหน
- ควรรอให้ราคายืนเหนือจุดไหนก่อน
- ตอนนี้ราคากำลังไล่ขึ้นเกินไปหรือยัง
- ตลาดเริ่มฟื้นพอให้เข้าแบบ `early recovery` ได้หรือยัง

ตัวอย่าง logic ที่ใช้ประกอบกัน:

- EMA9 / EMA21 เพื่อดูว่าตลาดเริ่มกลับมาแข็งแรงหรือยัง
- MACD bullish เพื่อคอนเฟิร์มแรงฟื้น
- volume ratio เพื่อกันการเด้งแบบ volume บาง
- support / resistance เพื่อคำนวณโซนรอเข้าและจุด breakout
- LSTM prediction เพื่อช่วยดูว่าควรรับแบบ pullback หรือเข้าตาม momentum

ผลคือบอทจะสามารถ:

- ซื้อได้เร็วขึ้นเมื่อโครงสร้างตลาดเริ่มดีขึ้นจริง
- ไม่ไล่ซื้อสูงเกินไปเมื่อราคาเริ่มวิ่ง
- แสดงโซนเข้าที่ AI มองไว้ใน Decision / wait hint ของ GUI

### 1.1 Dump Protection

นอกจาก downtrend guard แล้ว เวอร์ชันนี้จะไม่ยอมซื้อในช่วงที่ราคาโดนลากลงแรงระยะสั้น เช่น:

- แท่งล่าสุดแดงแรงผิดปกติ
- ราคาไหลลงต่อใน 1-3 แท่งล่าสุด
- แท่งปิดใกล้ low มาก บ่งชี้ว่ามีแรงขายกดต่อเนื่อง
- volume เร่งขึ้นพร้อมแรงขาย

เมื่อเข้าเงื่อนไขนี้ ระบบจะพัก BUY ชั่วคราว แม้ score ด้านอื่นจะดูดี เพื่อรอให้ราคา stabilize ก่อน

### 2. Adaptive Position Sizing

GUI และ console mode ใช้ risk manager คุมขนาดไม้แล้ว โดยจะลดขนาดไม้เมื่อ:

- ตลาดเป็น downtrend
- ATR สูงกว่าค่าที่กำหนด
- AI เอนลงแรง
- มี loss streak ต่อเนื่อง

ใน GUI ถึงแม้ช่อง `Auto Buy (THB)` จะใส่ไว้สูง ระบบก็ยัง cap ลงอัตโนมัติหาก risk regime ไม่เหมาะ

### 2.1 Profit Cashout: ถอนกำไร แต่คงเงินต้นในเหรียญ

เมื่อ position มีกำไรถึงระดับที่กำหนด ระบบสามารถเลือก `ขายเฉพาะกำไร` แทนการปิดทั้ง position ได้ โดย logic จะพยายามทำให้:

- เงินกำไรที่เกิดขึ้นถูกขายกลับมาเป็น THB
- มูลค่าเงินต้นเดิมยังคงถืออยู่ในเหรียญ
- stop loss / take profit รอบถัดไปถูกคำนวณใหม่จากเงินต้นที่ยังค้างในเหรียญ
- ถ้าราคาไปต่อ ระบบสามารถถอนกำไรเพิ่มได้อีกในรอบถัดไป

พฤติกรรมนี้จะถูกใช้กับจังหวะทำกำไร เช่น `TAKE_PROFIT` และ `AI TAKE PROFIT` แต่จะไม่ใช้กับ `STOP LOSS` หรือ `AI CUTLOSS` ที่ต้องการปิดความเสี่ยงจริง

### 2.2 Small Position Fee Guard

สำหรับไม้เล็ก ระบบจะไม่รีบขายขาดทุนเร็วเกินไปเพียงเพราะโดนค่าธรรมเนียมกินหรือเกิด loss ตื้น ๆ ช่วงสั้น ๆ โดยจะหน่วงการขายไว้ก่อนเมื่อ:

- มูลค่าไม้ยังเล็กอยู่
- ขาดทุนสุทธิยังอยู่ในช่วงใกล้กับ fee noise
- loss % ยังไม่ลึกพอถึงขั้นควร cut จริง

guard นี้จะช่วยลดการปิดไม้เล็กแบบเสียค่าธรรมเนียม 2 ฝั่งโดยยังไม่จำเป็น แต่ถ้าขาดทุนลึกถึง hard limit จริง ระบบยังคงยอมขายเพื่อลดความเสี่ยงเหมือนเดิม

### 3. Re-entry ที่ระวังขึ้น

Boss buy-back และ auto re-buy ปรับให้รอมากขึ้น:

- มี cooldown หลายรอบก่อนกลับเข้า
- ต้อง confirm หลายรอบเหนือ trigger
- มี trigger buffer เพิ่ม เพื่อกันการเด้งปลอม

## ค่าเริ่มต้นปัจจุบัน

### Trading Defaults

| ค่า | ค่าเริ่มต้น |
|---|---:|
| Interval | 30 วินาที |
| Stop Loss | 1.40% |
| Take Profit | 4.80% |
| AI CutLoss Review | 0.75% |
| AI Hard CutLoss | 1.65% |
| Profit Cashout | ON @ 1.60% |
| Small Position Fee Guard | ON |
| Buy Fee | 0.27% |
| Sell Fee | 0.27% |

### Risk Defaults

| ค่า | ค่าเริ่มต้น |
|---|---:|
| Max Trade Size | 7,000 THB |
| Max Position / Balance | 16% |
| Cash Reserve | 25% |
| Max Daily Loss | 3,000 THB |
| Max Daily Trades | 8 |
| Max Consecutive Losses | 3 |
| Downtrend Position Scale | 45% ของไม้ปกติ |
| High Volatility Scale | 70% ของไม้ปกติ |
| Pause Buying in Downtrend After Loss Streak | 2 ไม้ |

### GUI Runtime Defaults

| ค่า | ค่าเริ่มต้น |
|---|---:|
| Auto Buy | 100 THB |
| Boss CutLoss | 0.75% |
| Boss Recovery | 0.70% |
| Auto Re-Buy Rise | 0.70% |
| Auto Re-Buy Delay | 0.90% |
| Re-entry Cooldown | 2 รอบ |
| Re-entry Confirm | 3 รอบ |
| Profit Cashout Min Profit | 1.60% |

## การติดตั้ง

### 1. ติดตั้ง Python

แนะนำ Python 3.11+

### 2. สร้าง virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

### 3. ติดตั้ง dependencies

```powershell
pip install -r requirements.txt
```

### 4. ติดตั้ง PyTorch

CPU:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

CUDA 12.1:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## การตั้งค่า API

ใช้ environment variables หรือกรอกผ่าน GUI ก็ได้

```powershell
$env:BITKUB_API_KEY = "your_api_key"
$env:BITKUB_API_SECRET = "your_api_secret"
```

ข้อควรระวัง: เปิดสิทธิ์แค่ `Read` และ `Trade` และไม่ควรเปิด `Withdraw`

### จะเอา API Key / Secret / OpenAI Key มาจากไหน

#### 1. Bitkub API Key และ API Secret

ใช้สำหรับให้บอทอ่าน wallet, อ่านราคา และส่งคำสั่งซื้อขายไปที่ Bitkub

วิธีขอจาก Bitkub แบบทั่วไป:

1. สมัครและยืนยันตัวตนบัญชี Bitkub ให้เรียบร้อยก่อน
2. เข้าสู่ระบบบนเว็บทางการของ Bitkub
3. ไปที่หน้าโปรไฟล์หรือหน้าจัดการ API ของบัญชี
4. สร้าง API key ใหม่สำหรับบอทตัวนี้
5. กำหนดสิทธิ์เป็น `Read` และ `Trade` เท่านั้น
6. ห้ามเปิดสิทธิ์ `Withdraw`
7. ถ้า Bitkub มีตัวเลือกจำกัด IP ให้ใส่ IP ที่ใช้งานจริง จะปลอดภัยกว่า
8. คัดลอก `API Key` และ `API Secret` มาเก็บไว้ใช้งาน

ข้อสำคัญ:

- `API Secret` มักจะแสดงให้ดูได้ครั้งเดียวตอนสร้าง ควรเก็บทันที
- ถ้าหลุดหรือสงสัยว่ารั่ว ให้ลบ key เดิมแล้วสร้างใหม่
- ถ้าจะเริ่มทดสอบระบบ แนะนำเริ่มจาก `Paper Trading` ก่อน แม้จะมี key จริงแล้วก็ตาม

#### 2. OpenAI API Key

ใช้เมื่อเปิด `LLM Trade` หรือใช้คำแนะนำจาก LLM ในการช่วย review จังหวะเข้าออก

วิธีขอจาก OpenAI:

1. สมัครหรือเข้าสู่ระบบที่ `https://platform.openai.com/`
2. ไปที่หน้า `API Keys`
3. กดสร้าง secret key ใหม่
4. คัดลอก key มาเก็บทันที เพราะระบบอาจไม่แสดงซ้ำเต็มรูปแบบในภายหลัง
5. เติม billing / usage limit ให้เรียบร้อยถ้าบัญชียังไม่ได้เปิดใช้งาน API

ตัวอย่างการตั้งค่า:

```powershell
$env:OPENAI_API_KEY = "your_openai_api_key"
$env:LLM_ENABLED = "true"
```

ถ้าไม่ใช้ LLM:

- ไม่จำเป็นต้องมี OpenAI key
- บอทยังทำงานได้ด้วย indicator + LSTM + RL ตามปกติ

#### 3. key อื่น ๆ

เวอร์ชันปัจจุบันของแอพจำเป็นหลัก ๆ แค่:

- `BITKUB_API_KEY`
- `BITKUB_API_SECRET`
- `OPENAI_API_KEY` เฉพาะกรณีที่ต้องการใช้ LLM

## Runtime และ Environment ที่เกี่ยวกับ Profit Cashout

ค่าใหม่ที่เกี่ยวข้องกับการถอนกำไรมีดังนี้:

- `PROFIT_CASHOUT_ENABLED=true|false`
- `PROFIT_CASHOUT_MIN_PROFIT_PCT=1.6`
- `PROFIT_CASHOUT_MIN_THB=25`
- `SMALL_POSITION_FEE_GUARD_ENABLED=true|false`
- `SMALL_POSITION_FEE_GUARD_MAX_COST_THB=250`
- `SMALL_POSITION_FEE_GUARD_MIN_LOSS_BUFFER_THB=1.5`
- `SMALL_POSITION_FEE_GUARD_FEE_MULTIPLE=2.5`
- `SMALL_POSITION_FEE_GUARD_MAX_LOSS_PCT=1.8`

คำอธิบาย:

- `PROFIT_CASHOUT_ENABLED`: เปิด/ปิดโหมดถอนกำไรบางส่วน
- `PROFIT_CASHOUT_MIN_PROFIT_PCT`: ต้องมีกำไรอย่างน้อยกี่ % จึงเริ่มถอนกำไร
- `PROFIT_CASHOUT_MIN_THB`: กำไรขั้นต่ำเป็นเงินบาทก่อนที่ระบบจะยอมขายบางส่วน
- `SMALL_POSITION_FEE_GUARD_ENABLED`: เปิด/ปิดการหน่วงขายขาดทุนของไม้เล็ก
- `SMALL_POSITION_FEE_GUARD_MAX_COST_THB`: ใช้ guard นี้กับไม้ที่ต้นทุนไม่เกินกี่บาท
- `SMALL_POSITION_FEE_GUARD_MIN_LOSS_BUFFER_THB`: loss ขั้นต่ำแบบเงินบาทที่ยังมองว่าเป็น fee noise ได้
- `SMALL_POSITION_FEE_GUARD_FEE_MULTIPLE`: คูณ fee รวมเพื่อขยายช่วงกันขายเร็วเกินไป
- `SMALL_POSITION_FEE_GUARD_MAX_LOSS_PCT`: ถ้าขาดทุนเกิน % นี้จะไม่ถือว่าเป็น fee noise แล้ว

ใน GUI สามารถปรับ `Profit Cashout` และ `Profit %` ได้ทันทีระหว่างที่ auto trade กำลังทำงาน โดยระบบจะ apply ค่าล่าสุดให้อัตโนมัติ

ถ้าในอนาคตเพิ่มผู้ให้บริการ AI หรือ exchange อื่น ให้ขอ key จากหน้า developer หรือ dashboard ทางการของผู้ให้บริการนั้นเท่านั้น

### วิธีเก็บ key ให้ปลอดภัย

- อย่าเก็บ key ลงในไฟล์ `.txt`, source code, หรือรูปภาพในเครื่องแบบ plaintext
- อย่า commit key ลง Git
- ใช้ environment variables เป็นหลัก
- ถ้าต้องแชร์เครื่องกับคนอื่น ให้สร้าง key แยกและตั้งสิทธิ์ให้น้อยที่สุด
- ถ้าคิดว่า key รั่ว ให้ rotate ทันที

## วิธีรัน

### GUI

```powershell
python main.py gui
```

### Console trade

```powershell
python main.py trade
```

### Paper trade

```powershell
python main.py trade --paper
```

### Backtest

```powershell
python main.py backtest
```

### Train AI

```powershell
python main.py train
```

## การใช้ GUI แบบปัจจุบัน

### ภาพรวม layout

หน้าจอถูกแบ่งเป็น 2 ฝั่งหลัก:

- ฝั่งซ้าย: ใช้ดูสถานะตลาดและสถานะพอร์ตแบบต่อเนื่อง
- ฝั่งขวา: ใช้เชื่อมต่อ, ตั้งค่า, ควบคุมบอท และดูคำสั่งล่าสุด

รายละเอียดแต่ละฝั่ง:

| ฝั่ง | เนื้อหา |
|---|---|
| ซ้าย | `Market Data`, `สถานะและผลงานบอท`, `Real-Time P/L`, `Wallet`, `Technical Indicators`, `Open Positions`, `Activity Log` |
| ขวา | `Quick Start`, `API Settings`, `Quick Trade`, `AI Prediction`, `Controls`, `Risk Status`, `Trade History` |

การเลื่อนหน้าจอ:

- ถ้าเนื้อหาฝั่งซ้ายยาวเกินหน้าจอ สามารถเลื่อนเมาส์บนฝั่งซ้ายเพื่อ scroll ได้
- ถ้าเนื้อหาฝั่งขวายาวเกินหน้าจอ สามารถเลื่อนเมาส์บนฝั่งขวาเพื่อ scroll ได้
- ระบบจะเลื่อนเฉพาะ panel ที่เมาส์กำลังชี้อยู่
- สามารถลาก scrollbar ของแต่ละฝั่งได้โดยตรงเพื่อเลื่อนไปยังส่วนล่างได้เร็วขึ้น
- ถ้าอยู่ในคอลัมน์ที่การ์ดแน่นมาก ให้ใช้ `Shift` + ล้อเมาส์ เพื่อเลื่อนแบบเป็นหน้า
- ถ้าต้องการกระโดดไปส่วนบนหรือส่วนล่างของแต่ละฝั่งทันที ให้กด `TOP` หรือ `BOTTOM` เหนือ panel นั้น
- การ์ดฝั่งขวาที่ไม่ได้ใช้งานชั่วคราวสามารถกด `-` เพื่อยุบเก็บ และกด `+` เพื่อเปิดกลับได้

### 1. เชื่อมต่อ

- กรอก API Key / Secret
- เลือก symbol
- กด `Connect & Load Wallet`

หลังเปิดแอพ ให้เริ่มดูที่การ์ด `Quick Start` มุมขวาบนก่อน เพราะจะแสดง:

- สถานะการเชื่อมต่อ
- โหมดปัจจุบัน `LIVE / PAPER`
- สิ่งที่ควรทำต่อในขั้นตอนถัดไป

ถ้ายังไม่ได้เชื่อมต่อ การ์ดนี้จะบอกชัดว่าควรเริ่มจาก `Connect & Load Wallet` ก่อน

### 2. ตั้งค่าการเทรด

ในแผง `Controls` มีค่า runtime สำคัญดังนี้:

- `Interval`, `SL`, `TP`
- `Auto Buy (THB)`
- `AI Scale-In`, `AI Take Profit`
- `Paper Trading`
- `Boss Mode`, `Boss CutLoss`, `Boss Recovery`
- `Auto Re-Buy`, `Buy Up %`, `Delay Down %`

เมื่อบอทกำลังรันอยู่:

- ค่าที่แก้ใน `Controls` จะถูก apply เข้าระบบทันทีโดยไม่ต้องกด stop/start ใหม่
- badge จะเปลี่ยนจาก `PENDING` เป็น `LIVE` เมื่อค่าถูกนำไปใช้แล้ว
- ถ้ากรอกค่าผิด badge จะขึ้น `INVALID` และระบบจะใช้ค่าล่าสุดที่ถูกต้องต่อไป
- `Paper Trading` ยังเป็นข้อยกเว้น: ห้ามสลับระหว่างบอทรัน เพื่อไม่ให้พอร์ตจำลองปนกับพอร์ตจริง

หมายเหตุเพิ่มเติมระหว่าง auto trade:

- ถ้าระบบตรวจว่าราคาอยู่ในภาวะโดนลากลงแรง จะพัก BUY ทันทีแม้ AI หรือ indicator บางตัวเริ่มดูน่าสนใจ
- ถ้าต้องการปิดโปรแกรม ต้องกด `STOP BOT` ก่อนเสมอ ระบบจะไม่ยอมปิดหน้าต่างถ้าบอทยังรันอยู่

Layout ฝั่งขวาแบบใหม่:

- `Quick Start`: ดูภาพรวมและขั้นตอนถัดไป
- `API Settings`: เชื่อมต่อและเลือกเหรียญ
- `Quick Trade`: สั่งซื้อ/ขายเองแบบทันที
- `AI Prediction`: ดูมุมมอง AI ล่าสุด
- `Controls`: ตั้งค่าหลักของบอท
- `Risk Status`: ตรวจ market regime และ risk guard
- `Trade History`: ดูคำสั่งล่าสุด พร้อม badge เช่น `FEE-GUARD` และ note สั้น ๆ ว่าเหรียญไหนถูกระบบหน่วงขายเพราะ fee noise

ทุกการ์ดในฝั่งขวาสามารถยุบ/ขยายได้จากปุ่มที่หัวการ์ด และ `Trade History` จะถูกยุบไว้ก่อนเป็นค่าเริ่มต้นเพื่อประหยัดพื้นที่แนวตั้ง

โครง `Controls` แบ่งเป็น 3 ส่วนเพื่ออ่านง่ายขึ้น:

- `Core Settings`: Interval, SL, TP, Auto Buy
- `Automation`: AI Scale-In, AI Take Profit, Profit Cashout, Small Fee Guard, LLM Trade, Paper Trading
- `Recovery`: Boss Mode, Boss CutLoss, Boss Recovery, Auto Re-Buy

หมายเหตุ:

- ถ้าหน้าจอสูงไม่พอ สามารถเลื่อน sidebar ฝั่งขวาลงได้
- ถ้าต้องการดู log, positions, หรือ wallet เพิ่ม ให้เลื่อนฝั่งซ้ายลงได้เช่นกัน

หมายเหตุสำคัญ:

- ช่อง `Auto Buy (THB)` ไม่ได้แปลว่าระบบจะซื้อเต็มจำนวนทุกครั้ง
- risk manager จะ cap ไม้ให้เหมาะกับสภาพตลาดก่อนส่งคำสั่งจริง
- ถ้าอยู่ใน downtrend แรงหรือ ATR สูง ระบบอาจลดไม้จนต่ำกว่า 10 THB และข้ามรอบซื้อไปเลย

### 3. อ่านแผง `Risk Status`

ตอนนี้แผงนี้จะแสดงมากกว่า daily loss แบบเดิม:

- Daily loss / exposure / open positions
- Market regime เช่น `Normal`, `Downtrend Guard`, `High Volatility`, `Downtrend + Volatility`
- Position scale ที่ระบบใช้จริงในรอบนั้น
- หมายเหตุว่าโดน guard จากอะไร เช่น downtrend, ATR สูง, volume เบา, AI downside

การ์ดที่ควรดูคู่กันระหว่างใช้งาน:

- `Quick Start`: ดูว่าระบบคาดหวังให้ทำอะไรต่อ
- `สถานะและผลงานบอท`: ดู action ล่าสุดและเหตุผลการตัดสินใจ
- `Risk Status`: ดูว่าระบบกำลังลดไม้หรือพักซื้อหรือไม่
- `Activity Log`: ดูลำดับเหตุการณ์ละเอียดในแต่ละรอบ

เมื่อยังไม่เข้า BUY ให้ดู `Decision` เพิ่มเติม:

- ระบบจะแสดงโซนที่ AI อยากรอเข้า
- ถ้าราคายังไม่ผ่านจุด trigger ระบบจะบอกว่ารอให้ยืนเหนือราคาไหนก่อน
- ถ้าราคาเริ่มไล่สูงเกินไป ระบบจะบอกว่ารอให้ย่อกลับเข้าโซนที่เหมาะสม

ส่วนการ์ด `Quick Trade` จะถูกแยกบทบาทชัดเจนจาก `Auto Buy` ของบอท:

- `Manual Buy + Auto Trade`: ซื้อทันทีด้วยจำนวน THB ที่กรอก แล้วถ้าบอทยังไม่รัน ระบบจะเริ่ม auto trade ให้ดูแล position ต่อทันที
- `Auto Trade Only`: ไม่ส่งคำสั่งซื้อแบบ manual แต่ให้บอทประเมิน BUY หนึ่งรอบทันที โดยใช้กฎเดียวกับ auto trade จริง ทั้ง AI, LLM gate, risk sizing และ Auto Buy cap
- `Auto Buy` ใน `Controls` คือวงเงินอ้างอิงที่ระบบเอาไปคำนวณต่อกับ risk cap อีกชั้นสำหรับรอบ auto trade ปกติ

### 4. ลำดับใช้งานแบบสั้นที่สุด

ถ้าต้องการใช้งาน UI แบบง่ายที่สุด ให้ทำตามนี้:

1. เปิดแอพแล้วดู `Quick Start`
2. กรอก API และกด `Connect & Load Wallet`
3. ตั้ง `Paper Trading` หรือ `Live` ให้ถูกต้อง
4. ตรวจ `Interval`, `SL`, `TP`, `Auto Buy`
5. ถ้าต้องการสั่งซื้อเร็ว ให้เลือกโหมดใน `Quick Trade` ว่าจะ `Manual Buy + Auto Trade` หรือ `Auto Trade Only`
6. กด `START BOT`
7. ระหว่างรันให้ดู `Decision`, `Risk Status`, `Real-Time P/L`, และ `Activity Log`

### 5. Paper Trading

เปิดจาก GUI หรือ env vars:

```powershell
$env:PAPER_TRADE_ENABLED = "true"
$env:PAPER_TRADE_START_BALANCE_THB = "1000"
```

เมื่อเปิดแล้ว:

- จะไม่ส่งคำสั่งจริงไป Bitkub
- ใช้ยอดเงินจำลองแยกจาก wallet จริง
- position จำลองจะไม่ไปปะปนกับ holding จริง

## Environment Variables ที่ใช้บ่อย

```powershell
$env:TRADING_SYMBOL = "BTC_THB"
$env:STOP_LOSS_PCT = "1.4"
$env:TAKE_PROFIT_PCT = "4.8"
$env:PAPER_TRADE_ENABLED = "true"
$env:PAPER_TRADE_START_BALANCE_THB = "1000"
$env:BUY_FEE_RATE = "0.0027"
$env:SELL_FEE_RATE = "0.0027"

$env:MAX_TRADE_SIZE_THB = "7000"
$env:MAX_POSITION_PCT = "16"
$env:MAX_DAILY_LOSS_THB = "3000"
$env:MAX_DAILY_TRADES = "8"
$env:CASH_RESERVE_PCT = "25"

$env:DOWNTREND_POSITION_SCALE_PCT = "45"
$env:HIGH_VOLATILITY_ATR_PCT = "2.2"
$env:HIGH_VOLATILITY_POSITION_SCALE_PCT = "70"
$env:DOWNTREND_PAUSE_LOSS_STREAK = "2"

$env:LLM_ENABLED = "true"
$env:OPENAI_API_KEY = "<your_openai_api_key>"
$env:LLM_MODEL = "gpt-4.1-mini"
```

## หมายเหตุด้านกลยุทธ์

- ระบบนี้ยังไม่ใช่ hedge strategy และไม่ได้ short ตลาด
- เวลาตลาดลงแรง ระบบจะเน้น `ลดการเข้า`, `ลดไม้`, และ `รอ confirmation` มากกว่าฝืนซื้อถัว
- เวลาราคาโดนลากลงเร็ว ระบบ dump protection จะบล็อก BUY ชั่วคราวเพื่อลดโอกาสรับมีด
- backup stop loss ยังอยู่เพื่อกันกรณี AI ไม่ตอบสนองหรือสภาพตลาดผิดปกติ
- ผลลัพธ์จริงขึ้นกับค่าธรรมเนียม, slippage, latency และคุณภาพของข้อมูลจาก exchange

## Environment Variables เพิ่มเติมที่เกี่ยวกับ Dump Protection

```powershell
$env:DUMP_GUARD_ENABLED = "true"
$env:DUMP_SINGLE_CANDLE_DROP_PCT = "1.10"
$env:DUMP_THREE_CANDLE_DROP_PCT = "2.20"
$env:DUMP_NEAR_LOW_BUFFER_PCT = "0.35"
$env:DUMP_VOLUME_RATIO_MIN = "1.20"
```

## คำแนะนำก่อนใช้เงินจริง

1. เริ่มจาก `paper trading` ก่อน
2. ปรับ `Auto Buy`, `MAX_TRADE_SIZE_THB`, และ `MAX_POSITION_PCT` ให้เหมาะกับขนาดพอร์ต
3. ดูแผง `Risk Status` ว่าระบบกำลังลดไม้จากเหตุผลอะไรบ้าง
4. ถ้าตลาดแกว่งแรงผิดปกติ ให้ลด symbol scope หรือเพิ่ม interval ก่อน
