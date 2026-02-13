# 🚀 Loan Form OCR Pipeline

A lightweight document AI pipeline that converts structured loan application forms into clean, normalized CSV data.

Built using:

- 🖼 Template-based image cropping  
- 🔎 Field-level OCR  
- 🧠 Mistral LLM normalization  
- 📊 Structured CSV export  

---

## ✨ What It Does

Given a folder of loan form images:

1. Crops predefined fields using a blueprint
2. Runs OCR on each field
3. Sends extracted text to Mistral for cleanup + normalization
4. Outputs one clean CSV row per form

No layout models.  
No overengineering.  
Just deterministic structure + smart post-processing.



