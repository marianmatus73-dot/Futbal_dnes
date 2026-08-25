export const SPORTS = [
  "football",
  "tennis",
  "basketball",
  "hockey",
  "baseball",
  "mma",
  "nfl",
] as const;

export type Sport = (typeof SPORTS)[number];

export type Tip = {
  sport: Sport | string;
  league: string;
  match: string;
  pick: string;
  odds: number;
  model_probability: number;
  market_probability: number;
  edge: number;
  confidence: number;
  bookmaker: string;
  stake_amount: number;
  risk: string;
  release_stage?: "EARLY" | "FINAL" | string;
  lineup_verified?: boolean;
  reason?: string;
  raw_edge?: number;
  rejected_reasons?: string[];
  decision?: "ACCEPT" | "REJECT" | string;
  start_time?: string;
};

export type TipCard = {
  schema_version: number;
  generated_at: string;
  publishable: boolean;
  selected: Tip[];
  rejected_sample?: Tip[];
};

export type ModelRow = {
  sport: string;
  settled?: number;
  settled_bets?: number;
  yield?: number;
  yield_pct?: number;
  profit?: number;
  net_profit?: number;
  average_clv?: number | null;
  average_clv_pct?: number | null;
  brier_score?: number | null;
  max_drawdown?: number | null;
};

export type ModelTable = {
  generated_at?: string;
  sports?: ModelRow[] | Record<string, { all_time?: ModelRow }>;
  rows?: ModelRow[];
};

export type AppData = {
  tipCard: TipCard;
  modelRows: ModelRow[];
  source: "live" | "cache";
  refreshedAt: string;
};
