// 在 /api/scrape/route.ts 顶部添加
export const dynamic = "force-dynamic"; // 强制每次请求都动态处理

import axios from "axios";
import * as cheerio from "cheerio";
import { NextResponse } from "next/server";
import { HistoryEntry } from "@/types/lottery";

export async function GET() {
  try {
    const response = await axios.get(
      "https://datachart.500.com/ssq/history/history.shtml"
    );
    const $ = cheerio.load(response.data);

    const data: HistoryEntry[] = [];

    $("#tdata .t_tr1").each((_, row) => {
      const cells = $(row).find("td");
      const issue = $(cells[0]).text().trim();
      const reds = [];
      for (let i = 1; i <= 6; i++) {
        const num = parseInt($(cells[i]).text().trim(), 10);
        if (!isNaN(num)) reds.push(num);
      }
      const blue = parseInt($(cells[7]).text().trim(), 10);

      if (issue && reds.length === 6 && !isNaN(blue)) {
        data.push({ issue, reds, blue });
      }
    });

    console.log("Scraped Data:", data);

    return NextResponse.json({
      success: true,
      data,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error scraping data:", error);
    return NextResponse.json(
      { success: false, error: "Failed to fetch data" },
      { status: 500 }
    );
  }
}
