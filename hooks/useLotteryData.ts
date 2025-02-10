import { useState, useEffect } from "react";
import { HistoryEntry, ScrapeResponse } from "@/types/lottery";

export function useLotteryData() {
  const [historyData, setHistoryData] = useState<HistoryEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("/api/scrape");
        const result: ScrapeResponse = await response.json();

        if (!result.success || !result.data) {
          throw new Error(result.error || "Failed to fetch data");
        }

        setHistoryData(result.data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "An error occurred");
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  return { historyData, isLoading, error };
}
