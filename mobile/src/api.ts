import AsyncStorage from "@react-native-async-storage/async-storage";

import { DATA_BASE_URL } from "./config";
import type { AppData, MobileHistory, ModelTable, TipCard } from "./types";

const CACHE_KEY = "multisport:last-data:v1";

async function fetchJson<T>(name: string): Promise<T> {
  const response = await fetch(`${DATA_BASE_URL}/${name}`, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(`Server vrátil ${response.status}`);
  }
  return (await response.json()) as T;
}

async function fetchOptionalJson<T>(name: string, fallback: T): Promise<T> {
  try {
    return await fetchJson<T>(name);
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
    }));
  }
  if (Array.isArray(table.rows)) return table.rows;
  return [];
}

export async function loadAppData(): Promise<AppData> {
  try {
    const [tipCard, table, history] = await Promise.all([
      fetchJson<TipCard>("latest_tip_card.json"),
      fetchJson<ModelTable>("professional_model_table.json"),
      fetchOptionalJson<MobileHistory>("mobile_tip_history.json", {
        schema_version: 1,
        generated_at: "",
        sports: {},
      }),
    ]);
    const value: AppData = {
      tipCard,
      modelRows: rowsFrom(table),
      historyBySport: history.sports ?? {},
      source: "live",
      refreshedAt: new Date().toISOString(),
    };
    await AsyncStorage.setItem(CACHE_KEY, JSON.stringify(value));
    return value;
  } catch (error) {
    const cached = await AsyncStorage.getItem(CACHE_KEY);
    if (cached) {
      const value = JSON.parse(cached) as AppData;
      return { ...value, historyBySport: value.historyBySport ?? {}, source: "cache" };
    }
    throw error;
  }
}
