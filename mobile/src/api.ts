import AsyncStorage from "@react-native-async-storage/async-storage";

import { DATA_BASE_URL } from "./config";
import type { AppData, ModelTable, TipCard } from "./types";

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
    const [tipCard, table] = await Promise.all([
      fetchJson<TipCard>("latest_tip_card.json"),
      fetchJson<ModelTable>("professional_model_table.json"),
    ]);
    const value: AppData = {
      tipCard,
      modelRows: rowsFrom(table),
      source: "live",
      refreshedAt: new Date().toISOString(),
    };
    await AsyncStorage.setItem(CACHE_KEY, JSON.stringify(value));
    return value;
  } catch (error) {
    const cached = await AsyncStorage.getItem(CACHE_KEY);
    if (cached) {
      return { ...(JSON.parse(cached) as AppData), source: "cache" };
    }
    throw error;
  }
}

