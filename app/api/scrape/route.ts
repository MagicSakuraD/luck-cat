import { NextResponse } from "next/server";
import puppeteer from "puppeteer";
import { saveToCache, getFromCache } from "@/lib/cache";
import { HistoryEntry } from "@/types/lottery";

export async function GET() {
  try {
    // 检查缓存
    const cached = await getFromCache();
    if (cached && cached.timestamp) {
      const cacheAge = Date.now() - new Date(cached.timestamp).getTime();
      // 如果缓存不超过1小时，直接返回缓存数据
      if (cacheAge < 3600000) {
        return NextResponse.json({
          success: true,
          data: cached.data,
          timestamp: cached.timestamp,
          fromCache: true,
        });
      }
    }

    // 启动一个无头浏览器
    const browser = await puppeteer.launch({
      headless: true,
      args: [
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--disable-setuid-sandbox",
        "--no-first-run",
        "--no-sandbox",
        "--no-zygote",
        "--deterministic-fetch",
        "--disable-features=IsolateOrigins",
        "--disable-site-isolation-trials",
        "--ignore-certificate-errors",
        "--ignore-certificate-errors-spki-list",
        "--disable-http2", // 禁用 HTTP/2
      ],
    });

    const page = await browser.newPage();

    // 设置用户代理
    await page.setUserAgent(
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    );

    // 设置页面超时时间
    await page.setDefaultNavigationTimeout(60000);

    // 设置额外的请求头
    await page.setExtraHTTPHeaders({
      "Accept-Language": "zh-CN,zh;q=0.9",
      Accept:
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
      Connection: "keep-alive",
      "Cache-Control": "max-age=0",
    });

    // 访问数据页面
    await page.goto("https://datachart.500.com/ssq/history/history.shtml", {
      waitUntil: "networkidle0",
    });

    // 等待数据表格加载
    await page.waitForSelector("#tdata .t_tr1", { timeout: 60000 });

    // 提取数据
    const data: HistoryEntry[] = await page.evaluate(() => {
      const rows = document.querySelectorAll("#tdata .t_tr1");
      return Array.from(rows)
        .map((row) => {
          const cells = row.querySelectorAll("td");
          const issue = cells[0]?.innerText.trim();
          const reds = Array.from(cells)
            .slice(1, 7)
            .map((cell) => parseInt(cell.innerText.trim(), 10))
            .filter((num) => !isNaN(num));
          const blue = parseInt(cells[7]?.innerText.trim(), 10);

          return { issue, reds, blue };
        })
        .filter(
          (item) => item.issue && item.reds.length === 6 && !isNaN(item.blue)
        );
    });

    await browser.close();

    // 缓存数据到本地文件
    // 保存到缓存
    await saveToCache(data);

    return NextResponse.json({
      success: true,
      data,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error scraping data:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to fetch lottery data",
      },
      { status: 500 }
    );
  }
}
