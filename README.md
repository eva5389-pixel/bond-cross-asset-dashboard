# 全球跨資產優化與債券風險監控

Streamlit 儀表板，整合：

- 合作金庫 MoneyDJ 與 MoneyDJ 公開債券資料
- 依 ISIN 合併去重與來源標示
- 投資等級、非投資等級及新興市場債分類
- 殖利率、年限、產業、信評與溢折價篩選
- 基金／ETF 池與跨資產配置功能

## 執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

公開參考價格不等於保證成交價格。來源未提供 CDS 時，系統不會補造 CDS 數值。
