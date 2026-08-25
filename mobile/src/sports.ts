import type { Sport } from "./types";

export const sportMeta: Record<Sport, { label: string; icon: string; color: string }> = {
  football: { label: "Futbal", icon: "⚽", color: "#62E6A6" },
  tennis: { label: "Tenis", icon: "🎾", color: "#D8EE69" },
  basketball: { label: "Basketbal", icon: "🏀", color: "#FFAA64" },
  hockey: { label: "Hokej", icon: "🏒", color: "#73D0FF" },
  baseball: { label: "Baseball", icon: "⚾", color: "#FF8796" },
  mma: { label: "MMA", icon: "🥊", color: "#C59CFF" },
  nfl: { label: "NFL", icon: "🏈", color: "#D7A873" },
};

