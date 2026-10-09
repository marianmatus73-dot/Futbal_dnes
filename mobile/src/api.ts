import AsyncStorage from "@react-native-async-storage/async-storage";

import { DATA_BASE_URL } from "./config";
import type { AppData, MobileHistory, MobilePerformance, ModelGovernance, ModelTable, OperationalHealth, TipCard } from "./types";

const CACHE_KEY = "multisport:last-data:v1";

async function fetchJson<T>(name: string, refreshToken: string): Promise<T> {
  const response = await fetch(`${DATA_BASE_URL}/${name}?refresh=${refreshToken}`, {
    headers: {
      Accept: "application/json",
      "Cache-Control": "no-cache, no-store, must-revalidate",
      Pragma: "no-cache",
    },
  });
  if (!response.ok) {
    throw new Error(`Server vrátil ${response.status}`);
  }
  return (await response.json()) as T;
}

async function fetchOptionalJson<T>(name: string, fallback: T, refreshToken: string): Promise<T> {
  try {
    return await fetchJson<T>(name, refreshToken);
  } catch {
    return fallback;
  }
}

function rowsFrom(table: ModelTable) {
  if (Array.isArray(table.sports)) return table.sports;
  if (table.sports && typeof table.sports === "object") {
    return Object.entries(table.sports).map(([sport, value]) => ({
      sport,
      ...(value.all_time ?? {}),
      low_odds_1_20_1_60: value.low_odds_1_20_1_60,
    }));
  }
  if (Array.isArray(table.rows)) return table.rows;
  return [];
}

export async function loadAppData(): Promise<AppData> {
  try {
    // GitHub Raw is fronted by a CDN. A unique token prevents a successful
    // refresh from receiving yesterday's JSON from an HTTP cache.
    const refreshToken = Date.now().toString();
    const [tipCard, table, history, performance, operationalHealth, modelGovernance] = await Promise.all([
      fetchJson<TipCard>("latest_tip_card.json", refreshToken),
      // Only the current tip card is essential. A temporary failure of a
      // statistics file must not make the app replace fresh tips with its
      // entire stale AsyncStorage snapshot.
      fetchOptionalJson<ModelTable>("professional_model_table.json", {}, refreshToken),
      fetchOptionalJson<MobileHistory>("mobile_tip_history.json", {
        schema_version: 1,
        generated_at: "",
        sports: {},
      }, refreshToken),
      fetchOptionalJson<MobilePerformance>("mobile_performance.json", {
        schema_version: 1, generated_at: "", starting_bankroll: 1000,
        current_bankroll: 1000, points: [],
      }, refreshToken),
      fetchOptionalJson<OperationalHealth | undefined>("operational_health.json", undefined, refreshToken),
      fetchOptionalJson<ModelGovernance | undefined>("model_governance.json", undefined, refreshToken),
    ]);
    const value: AppData = {
      tipCard,
      modelRows: rowsFrom(table),
      handballShadow: table.shadow_models?.handball,
      historyBySport: history.sports ?? {},
      resultsBySport: history.results_sports ?? {},
      performance,
      operationalHealth,
      modelGovernance,
      source: "live",
      refreshedAt: new Date().toISOString(),
    };
    await AsyncStorage.setItem(CACHE_KEY, JSON.stringify(value));
    return value;
  } catch (error) {
    const cached = await AsyncStorage.getItem(CACHE_KEY);
    if (cached) {
      const value = JSON.parse(cached) as AppData;
      return { ...value, historyBySport: value.historyBySport ?? {}, resultsBySport: value.resultsBySport ?? {}, performance: value.performance ?? { schema_version: 1, generated_at: "", starting_bankroll: 1000, current_bankroll: 1000, points: [] }, source: "cache" };
    }
    throw error;
  }
}

