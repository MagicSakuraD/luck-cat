import fs from "fs/promises";
import path from "path";
import { HistoryEntry } from "@/types/lottery";

const CACHE_DIR = path.join(process.cwd(), "cache");
const CACHE_FILE = path.join(CACHE_DIR, "lottery-data.json");

export async function saveToCache(data: HistoryEntry[]) {
  try {
    await fs.mkdir(CACHE_DIR, { recursive: true });
    await fs.writeFile(
      CACHE_FILE,
      JSON.stringify({ data, timestamp: new Date().toISOString() }, null, 2)
    );
  } catch (error) {
    console.error("Error saving to cache:", error);
  }
}

export async function getFromCache() {
  try {
    const data = await fs.readFile(CACHE_FILE, "utf-8");
    return JSON.parse(data);
  } catch {
    return null;
  }
}
