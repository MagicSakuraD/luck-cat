export interface HistoryEntry {
  issue: string;
  reds: number[];
  blue: number;
}

export interface ScrapeResponse {
  success: boolean;
  data?: HistoryEntry[];
  error?: string;
  timestamp?: string;
}
