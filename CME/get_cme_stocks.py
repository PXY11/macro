import cloudscraper
import pandas as pd
import os
from datetime import datetime

# CME 正确 URL（包含 Silver 的特例 stocks）
METAL_URLS = {
    "Gold":      "https://www.cmegroup.com/delivery_reports/Gold_Stocks.xls",
    "Silver":    "https://www.cmegroup.com/delivery_reports/Silver_stocks.xls",   # 唯一小写
    "Copper":    "https://www.cmegroup.com/delivery_reports/Copper_Stocks.xls",
    "Platinum_Palladium": 
                 "https://www.cmegroup.com/delivery_reports/PA-PL_Stck_Rprt.xls",
    "Aluminum":  "https://www.cmegroup.com/delivery_reports/Aluminum_Stocks.xls",
    "Zinc":      "https://www.cmegroup.com/delivery_reports/Zinc_Stocks.xls",
    "Lead":      "https://www.cmegroup.com/delivery_reports/Lead_Stocks.xls",
}

def parse_report_date_from_excel(path):
    df = pd.read_excel(path, header=None)
    for row in df.values:
        for cell in row:
            if isinstance(cell, str) and "Report Date" in cell:
                date_str = cell.split(":")[1].strip()
                return datetime.strptime(date_str, "%m/%d/%Y")
    raise ValueError(f"{path}: 无法找到 Report Date")

def safe_rename(src, dst):
    if os.path.exists(dst):
        print(f"⚠️ 已存在，删除旧文件：{dst}")
        os.remove(dst)
    os.rename(src, dst)

def download_all_metals(save_root="./CME_Stocks", proxy_port=7890):
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )

    proxies = {
        "http":  f"http://127.0.0.1:{proxy_port}",
        "https": f"http://127.0.0.1:{proxy_port}",
    }

    os.makedirs(save_root, exist_ok=True)

    for metal, url in METAL_URLS.items():
        print(f"\n=== 下载 {metal} ===")

        metal_dir = os.path.join(save_root, metal)
        os.makedirs(metal_dir, exist_ok=True)

        tmp_path = os.path.join(metal_dir, f"{metal}_tmp.xls")

        print("Downloading:", url)
        r = scraper.get(url, proxies=proxies, timeout=120)
        r.raise_for_status()

        with open(tmp_path, "wb") as f:
            f.write(r.content)

        try:
            date_obj = parse_report_date_from_excel(tmp_path)
        except Exception as e:
            print("❌ 日期解析失败：", e)
            continue

        date_str = date_obj.strftime("%Y%m%d")
        final_path = os.path.join(metal_dir, f"{metal}_Stocks_{date_str}.xls")

        safe_rename(tmp_path, final_path)
        print(f"✔ 保存成功：{final_path}")

    print("\n🎉 全部金属已下载完成！")

# 运行
download_all_metals(proxy_port=7890)
