import { StatusBar } from "expo-status-bar";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaProvider, SafeAreaView, useSafeAreaInsets } from "react-native-safe-area-context";

import { loadAppData } from "./src/api";
import { REFRESH_INTERVAL_MS } from "./src/config";
import { sportMeta } from "./src/sports";
import { colors } from "./src/theme";
import { SPORTS, type AppData, type ModelRow, type Sport, type Tip } from "./src/types";

type Screen = "today" | "sports" | "performance" | "system";

const emptyCard = {
  schema_version: 2,
  generated_at: "",
  publishable: false,
  selected: [],
};

const formatPercent = (value?: number | null) =>
  value == null ? "—" : `${(Math.abs(value) <= 1 ? value * 100 : value).toFixed(1)} %`;

const formatNumber = (value?: number | null) =>
  value == null ? "—" : Number(value).toFixed(2);

const relativeTime = (iso: string) => {
  if (!iso) return "zatiaľ bez behu";
  const minutes = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000));
  if (minutes < 2) return "práve teraz";
  if (minutes < 60) return `pred ${minutes} min`;
  return `pred ${Math.round(minutes / 60)} h`;
};

const formatStartTime = (iso?: string) => {
  if (!iso) return "";
  const value = new Date(iso);
  if (Number.isNaN(value.getTime())) return "";
  return value.toLocaleString("sk-SK", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
};

const riskColor = (risk?: string) => risk === "high" ? colors.danger : risk === "low" ? colors.primary : colors.warning;

function TipItem({ tip }: { tip: Tip }) {
  const meta = sportMeta[(tip.sport as Sport)] ?? sportMeta.football;
  const isFinal = tip.release_stage === "FINAL";
  return (
    <View style={styles.tipCard}>
      <View style={styles.tipHeader}>
        <View style={[styles.sportIcon, { backgroundColor: `${meta.color}22` }]}>
          <Text style={styles.sportEmoji}>{meta.icon}</Text>
        </View>
        <View style={styles.tipHeading}>
          <Text style={styles.league}>{tip.league}</Text>
          <Text style={styles.match}>{tip.match}</Text>
          {tip.start_time ? <Text style={styles.startTime}>{formatStartTime(tip.start_time)}</Text> : null}
        </View>
        <View style={[styles.stage, isFinal && styles.stageFinal]}>
          <Text style={[styles.stageText, isFinal && styles.stageFinalText]}>
            {isFinal ? "FINAL" : tip.release_stage || "EARLY"}
          </Text>
        </View>
      </View>

      <View style={styles.pickRow}>
        <View>
          <Text style={styles.label}>TIP</Text>
          <Text style={styles.pick}>{tip.pick}</Text>
        </View>
        <View style={styles.oddsBox}>
          <Text style={styles.label}>KURZ</Text>
          <Text style={styles.odds}>{tip.odds.toFixed(2)}</Text>
        </View>
      </View>

      <View style={styles.metrics}>
        <Metric label="Pravdepodobnosť" value={formatPercent(tip.model_probability)} />
        <Metric label="Edge" value={formatPercent(tip.edge)} positive />
        <Metric label="Confidence" value={`${Math.round(tip.confidence)}/100`} />
      </View>
      <View style={styles.tipFooter}>
        <Text style={styles.bookmaker}>{tip.bookmaker}</Text>
        <View style={styles.footerRight}><Text style={[styles.riskText, { color: riskColor(tip.risk) }]}>{tip.risk?.toUpperCase()}</Text><Text style={styles.stake}>Vklad {formatNumber(tip.stake_amount)}</Text></View>
      </View>
    </View>
  );
}

function Metric({ label, value, positive = false }: { label: string; value: string; positive?: boolean }) {
  return (
    <View style={styles.metric}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={[styles.metricValue, positive && { color: colors.primary }]}>{value}</Text>
    </View>
  );
}

function EmptyTips({ sport }: { sport?: Sport }) {
  const meta = sport ? sportMeta[sport] : undefined;
  return (
    <View style={styles.emptyCard}>
      <Text style={styles.emptyIcon}>{meta?.icon ?? "✓"}</Text>
      <Text style={styles.emptyTitle}>Žiadny potvrdený tip</Text>
      <Text style={styles.emptyText}>
        {meta ? `${meta.label} dnes zatiaľ nemá tip, ktorý prešiel profesionálnym filtrom.` : "NO BET je správne rozhodnutie, keď trh neponúka dostatočnú výhodu."}
      </Text>
    </View>
  );
}

function CandidateItem({ tip }: { tip: Tip }) {
  const [expanded, setExpanded] = useState(false);
  const meta = sportMeta[(tip.sport as Sport)] ?? sportMeta.football;
  const reasons = tip.rejected_reasons?.length
    ? tip.rejected_reasons
    : [tip.risk === "high" ? "Vysoké riziko" : "Neprešiel profesionálnym filtrom"];
  return (
    <View style={styles.candidateCard}>
      <View style={styles.tipHeader}>
        <View style={[styles.sportIcon, { backgroundColor: `${meta.color}22` }]}><Text style={styles.sportEmoji}>{meta.icon}</Text></View>
        <View style={styles.tipHeading}><Text style={styles.league}>{tip.league}</Text><Text style={styles.match}>{tip.match}</Text>{tip.start_time ? <Text style={styles.startTime}>{formatStartTime(tip.start_time)}</Text> : null}</View>
        <View style={styles.rejectedBadge}><Text style={styles.rejectedBadgeText}>NEVKLADAŤ</Text></View>
      </View>
      <View style={styles.pickRow}>
        <View><Text style={styles.label}>KANDIDÁT</Text><Text style={styles.pick}>{tip.pick}</Text></View>
        <View style={styles.oddsBox}><Text style={styles.label}>KURZ</Text><Text style={styles.odds}>{tip.odds.toFixed(2)}</Text></View>
      </View>
      <View style={styles.metrics}>
        <Metric label="Model" value={formatPercent(tip.model_probability)} />
        <Metric label="Consensus edge" value={formatPercent(tip.edge)} />
        <Metric label="Confidence" value={`${Math.round(tip.confidence)}/100`} />
      </View>
      <View style={styles.candidateFooter}><Text style={[styles.riskText, { color: riskColor(tip.risk) }]}>RIZIKO {tip.risk?.toUpperCase()}</Text><Pressable onPress={() => setExpanded((value) => !value)} hitSlop={10}><Text style={styles.detailButton}>{expanded ? "Skryť detail" : "Prečo neprešiel?"}</Text></Pressable></View>
      {expanded ? <View style={styles.rejectionBox}><Text style={styles.rejectionTitle}>Rozhodnutie profesionálneho filtra</Text>{reasons.map((reason, index) => <Text key={`${reason}-${index}`} style={styles.rejectionText}>• {reason}</Text>)}{tip.reason ? <Text style={styles.modelReason}>{tip.reason}</Text> : null}</View> : null}
    </View>
  );
}

function Candidates({ tips }: { tips: Tip[] }) {
  if (!tips.length) return null;
  return (
    <View style={styles.candidatesSection}>
      <Text style={styles.candidatesTitle}>Kandidáti na sledovanie</Text>
      <Text style={styles.candidatesSubtitle}>Zaujímavé možnosti z analýzy, ktoré neprešli bezpečnostným filtrom. Nie sú to potvrdené tipy.</Text>
      {tips.map((tip, index) => <CandidateItem key={`candidate-${tip.sport}-${tip.match}-${index}`} tip={tip} />)}
    </View>
  );
}

function Today({ data }: { data: AppData }) {
  const [filter, setFilter] = useState<"all" | "confirmed" | "candidates">("all");
  const tips = data.tipCard.selected ?? [];
  const candidates = data.tipCard.rejected_sample ?? [];
  return (
    <>
      <View style={styles.hero}>
        <Text style={styles.eyebrow}>DNES</Text>
        <Text style={styles.heroTitle}>{tips.length ? `${tips.length} kvalitné tipy` : "Pokojný deň. NO BET."}</Text>
        <Text style={styles.heroText}>Posledná analýza {relativeTime(data.tipCard.generated_at)}</Text>
      </View>
      <View style={styles.counters}><Summary label="Potvrdené" value={String(tips.length)} positive /><Summary label="Kandidáti" value={String(candidates.length)} /><Summary label="Športy" value={String(new Set([...tips, ...candidates].map((tip) => tip.sport)).size)} /></View>
      <View style={styles.filters}>
        {([['all', 'Všetko'], ['confirmed', 'Potvrdené'], ['candidates', 'Kandidáti']] as const).map(([value, label]) => <Pressable key={value} onPress={() => setFilter(value)} style={[styles.filterChip, filter === value && styles.filterChipActive]}><Text style={[styles.filterText, filter === value && styles.filterTextActive]}>{label}</Text></Pressable>)}
      </View>
      {filter !== "candidates" ? (tips.length ? tips.map((tip, index) => <TipItem key={`${tip.sport}-${tip.match}-${index}`} tip={tip} />) : <EmptyTips />) : null}
      {filter !== "confirmed" ? <Candidates tips={candidates} /> : null}
    </>
  );
}

function Sports({ data, selected, onSelect }: { data: AppData; selected: Sport; onSelect: (sport: Sport) => void }) {
  const tips = data.tipCard.selected.filter((tip) => tip.sport === selected);
  const candidates = (data.tipCard.rejected_sample ?? []).filter((tip) => tip.sport === selected);
  const row = data.modelRows.find((item) => item.sport === selected);
  return (
    <>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.sportTabs}>
        {SPORTS.map((sport) => {
          const meta = sportMeta[sport];
          const active = sport === selected;
          return (
            <Pressable key={sport} onPress={() => onSelect(sport)} style={[styles.sportTab, active && { borderColor: meta.color, backgroundColor: `${meta.color}18` }]}>
              <Text style={styles.tabEmoji}>{meta.icon}</Text>
              <Text style={[styles.sportTabText, active && { color: meta.color }]}>{meta.label}</Text>
            </Pressable>
          );
        })}
      </ScrollView>
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>{sportMeta[selected].label}</Text>
        <Text style={styles.sectionSubtitle}>Model a dnešné rozhodnutia</Text>
      </View>
      <View style={styles.summaryGrid}>
        <Summary label="Uzavreté" value={String(row?.settled_bets ?? row?.settled ?? 0)} />
        <Summary label="Yield" value={formatPercent(row?.yield_pct ?? row?.yield)} positive />
        <Summary label="Zisk" value={formatNumber(row?.net_profit ?? row?.profit)} />
      </View>
      {tips.length ? tips.map((tip, index) => <TipItem key={`${tip.match}-${index}`} tip={tip} />) : <EmptyTips sport={selected} />}
      <Candidates tips={candidates} />
    </>
  );
}

function Summary({ label, value, positive = false }: { label: string; value: string; positive?: boolean }) {
  return (
    <View style={styles.summary}>
      <Text style={styles.summaryLabel}>{label}</Text>
      <Text style={[styles.summaryValue, positive && { color: colors.primary }]}>{value}</Text>
    </View>
  );
}

function Performance({ rows }: { rows: ModelRow[] }) {
  return (
    <>
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>Výkon modelov</Text>
        <Text style={styles.sectionSubtitle}>Iba uzavreté tipy a skutočné výsledky</Text>
      </View>
      {SPORTS.map((sport) => {
        const row = rows.find((item) => item.sport === sport);
        const meta = sportMeta[sport];
        return (
          <View key={sport} style={styles.performanceRow}>
            <View style={[styles.sportIcon, { backgroundColor: `${meta.color}22` }]}><Text style={styles.sportEmoji}>{meta.icon}</Text></View>
            <View style={styles.performanceName}>
              <Text style={styles.performanceTitle}>{meta.label}</Text>
              <Text style={styles.performanceSub}>{row?.settled_bets ?? row?.settled ?? 0} uzavretých</Text>
            </View>
            <View style={styles.performanceMetric}><Text style={styles.label}>YIELD</Text><Text style={{ color: colors.primary, fontWeight: "800" }}>{formatPercent(row?.yield_pct ?? row?.yield)}</Text></View>
            <View style={styles.performanceMetric}><Text style={styles.label}>ZISK</Text><Text style={styles.performanceValue}>{formatNumber(row?.net_profit ?? row?.profit)}</Text></View>
          </View>
        );
      })}
    </>
  );
}

function System({ data }: { data: AppData }) {
  const ageMinutes = data.tipCard.generated_at ? (Date.now() - new Date(data.tipCard.generated_at).getTime()) / 60000 : Infinity;
  const healthy = ageMinutes < 24 * 60;
  return (
    <>
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>Stav systému</Text>
        <Text style={styles.sectionSubtitle}>Kontrola dát bez technických detailov</Text>
      </View>
      <StatusRow label="Tipovací engine" value={healthy ? "Aktívny" : "Dáta sú staršie"} ok={healthy} />
      <StatusRow label="Zdroj v aplikácii" value={data.source === "live" ? "Aktuálne online dáta" : "Posledná uložená verzia"} ok={data.source === "live"} />
      <StatusRow label="Posledný report" value={relativeTime(data.tipCard.generated_at)} ok={healthy} />
      <StatusRow label="Mobilné dáta" value="API kľúče nie sú v telefóne" ok />
      <View style={styles.infoCard}><Text style={styles.infoTitle}>Ako čítať tipy</Text><Text style={styles.infoText}>EARLY znamená skorý value kurz. FINAL znamená, že tip bol potvrdený blízko výkopu po dostupných zostavách. Ak nie je výhoda dostatočná, aplikácia zobrazí NO BET.</Text></View>
    </>
  );
}

function StatusRow({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return <View style={styles.statusRow}><View style={[styles.statusDot, { backgroundColor: ok ? colors.primary : colors.warning }]} /><View style={styles.statusText}><Text style={styles.statusLabel}>{label}</Text><Text style={styles.statusValue}>{value}</Text></View></View>;
}

function AppContent() {
  const insets = useSafeAreaInsets();
  const [screen, setScreen] = useState<Screen>("today");
  const [sport, setSport] = useState<Sport>("football");
  const [data, setData] = useState<AppData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async (silent = false) => {
    if (!silent) setRefreshing(true);
    try {
      setData(await loadAppData());
      setError("");
    } catch {
      setError("Dáta sa nepodarilo načítať. Skontroluj internet a skús znova.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    refresh(true);
    const timer = setInterval(() => refresh(true), REFRESH_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [refresh]);

  const title = useMemo(() => ({ today: "Môj Multisport", sports: "Športy", performance: "Štatistiky", system: "Systém" })[screen], [screen]);

  if (loading) return <SafeAreaView style={styles.loading}><StatusBar style="light" /><ActivityIndicator color={colors.primary} size="large" /><Text style={styles.loadingText}>Načítavam najnovšiu analýzu…</Text></SafeAreaView>;

  const usable = data ?? { tipCard: emptyCard, modelRows: [], source: "live" as const, refreshedAt: "" };
  return (
    <SafeAreaView style={styles.safe} edges={["top", "left", "right"]}>
      <StatusBar style="light" />
      <View style={styles.topBar}><View><Text style={styles.brand}>MBE</Text><Text style={styles.topTitle}>{title}</Text></View><View style={styles.liveBadge}><View style={styles.liveDot} /><Text style={styles.liveText}>{usable.source === "live" ? "LIVE" : "OFFLINE"}</Text></View></View>
      <ScrollView style={styles.content} contentContainerStyle={styles.contentInner} refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => refresh()} tintColor={colors.primary} colors={[colors.primary]} />}>
        {error ? <Text style={styles.error}>{error}</Text> : null}
        {screen === "today" && <Today data={usable} />}
        {screen === "sports" && <Sports data={usable} selected={sport} onSelect={setSport} />}
        {screen === "performance" && <Performance rows={usable.modelRows} />}
        {screen === "system" && <System data={usable} />}
      </ScrollView>
      <View style={[styles.nav, { paddingBottom: Math.max(insets.bottom, 12) }]}>
        <NavItem icon="⌂" label="Dnes" active={screen === "today"} onPress={() => setScreen("today")} />
        <NavItem icon="◉" label="Športy" active={screen === "sports"} onPress={() => setScreen("sports")} />
        <NavItem icon="↗" label="Výkon" active={screen === "performance"} onPress={() => setScreen("performance")} />
        <NavItem icon="●" label="Systém" active={screen === "system"} onPress={() => setScreen("system")} />
      </View>
    </SafeAreaView>
  );
}

export default function App() {
  return <SafeAreaProvider><AppContent /></SafeAreaProvider>;
}

function NavItem({ icon, label, active, onPress }: { icon: string; label: string; active: boolean; onPress: () => void }) {
  return <Pressable onPress={onPress} style={styles.navItem}><Text style={[styles.navIcon, active && styles.navActive]}>{icon}</Text><Text style={[styles.navLabel, active && styles.navActive]}>{label}</Text></Pressable>;
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.background }, loading: { flex: 1, backgroundColor: colors.background, alignItems: "center", justifyContent: "center", gap: 16 }, loadingText: { color: colors.muted, fontSize: 15 },
  topBar: { paddingHorizontal: 20, paddingTop: 16, paddingBottom: 14, flexDirection: "row", justifyContent: "space-between", alignItems: "flex-end" }, brand: { color: colors.primary, fontWeight: "900", letterSpacing: 2, fontSize: 12 }, topTitle: { color: colors.text, fontWeight: "900", fontSize: 26, marginTop: 2 },
  liveBadge: { flexDirection: "row", alignItems: "center", borderWidth: 1, borderColor: colors.border, borderRadius: 20, paddingVertical: 7, paddingHorizontal: 10, gap: 7 }, liveDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: colors.primary }, liveText: { color: colors.muted, fontWeight: "800", fontSize: 10, letterSpacing: 1 },
  content: { flex: 1 }, contentInner: { paddingHorizontal: 16, paddingBottom: 30 }, error: { color: colors.danger, backgroundColor: "#3B1D2A", padding: 12, borderRadius: 12, marginBottom: 12 },
  hero: { backgroundColor: colors.primaryDark, borderWidth: 1, borderColor: "#24594D", borderRadius: 24, padding: 22, marginBottom: 16 }, eyebrow: { color: colors.primary, fontWeight: "900", fontSize: 11, letterSpacing: 1.8 }, heroTitle: { color: colors.text, fontWeight: "900", fontSize: 25, marginTop: 8 }, heroText: { color: "#A8CFC3", marginTop: 8, fontSize: 14 },
  tipCard: { backgroundColor: colors.surface, borderRadius: 22, borderWidth: 1, borderColor: colors.border, padding: 17, marginBottom: 14 }, tipHeader: { flexDirection: "row", alignItems: "center" }, sportIcon: { width: 43, height: 43, borderRadius: 14, alignItems: "center", justifyContent: "center" }, sportEmoji: { fontSize: 21 }, tipHeading: { flex: 1, marginHorizontal: 11 }, league: { color: colors.muted, fontSize: 11, textTransform: "uppercase", fontWeight: "800" }, match: { color: colors.text, fontSize: 15, fontWeight: "800", marginTop: 3 }, startTime: { color: colors.accent, fontSize: 10, fontWeight: "700", marginTop: 4 },
  stage: { backgroundColor: "#263751", paddingHorizontal: 9, paddingVertical: 6, borderRadius: 9 }, stageFinal: { backgroundColor: colors.primary }, stageText: { color: colors.muted, fontSize: 9, fontWeight: "900" }, stageFinalText: { color: colors.background },
  pickRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "flex-end", marginTop: 21, marginBottom: 18 }, label: { color: colors.muted, fontSize: 9, fontWeight: "900", letterSpacing: 1 }, pick: { color: colors.text, fontSize: 20, fontWeight: "900", marginTop: 4, maxWidth: 245 }, oddsBox: { alignItems: "flex-end" }, odds: { color: colors.warning, fontSize: 24, fontWeight: "900", marginTop: 2 },
  metrics: { flexDirection: "row", backgroundColor: colors.surfaceAlt, borderRadius: 15, padding: 12 }, metric: { flex: 1 }, metricLabel: { color: colors.muted, fontSize: 9, marginBottom: 4 }, metricValue: { color: colors.text, fontWeight: "800", fontSize: 14 }, tipFooter: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginTop: 13 }, bookmaker: { color: colors.muted, fontSize: 12 }, footerRight: { flexDirection: "row", gap: 10, alignItems: "center" }, riskText: { fontWeight: "900", fontSize: 9, letterSpacing: .5 }, stake: { color: colors.accent, fontWeight: "800", fontSize: 12 },
  emptyCard: { backgroundColor: colors.surface, borderWidth: 1, borderColor: colors.border, borderRadius: 22, padding: 28, alignItems: "center" }, emptyIcon: { fontSize: 32 }, emptyTitle: { color: colors.text, fontWeight: "900", fontSize: 18, marginTop: 12 }, emptyText: { color: colors.muted, textAlign: "center", lineHeight: 21, marginTop: 8 },
  counters: { flexDirection: "row", gap: 8, marginBottom: 12 }, filters: { flexDirection: "row", gap: 8, marginBottom: 16 }, filterChip: { borderWidth: 1, borderColor: colors.border, borderRadius: 16, paddingHorizontal: 13, paddingVertical: 8, backgroundColor: colors.surface }, filterChipActive: { borderColor: colors.primary, backgroundColor: colors.primaryDark }, filterText: { color: colors.muted, fontWeight: "800", fontSize: 11 }, filterTextActive: { color: colors.primary },
  candidatesSection: { marginTop: 22 }, candidatesTitle: { color: colors.warning, fontSize: 21, fontWeight: "900" }, candidatesSubtitle: { color: colors.muted, lineHeight: 19, marginTop: 5, marginBottom: 13 }, candidateCard: { backgroundColor: "#171C2B", borderRadius: 22, borderWidth: 1, borderColor: "#664D24", padding: 17, marginBottom: 14 }, rejectedBadge: { backgroundColor: "#4A2B20", borderWidth: 1, borderColor: colors.warning, borderRadius: 9, paddingHorizontal: 8, paddingVertical: 6 }, rejectedBadgeText: { color: colors.warning, fontSize: 8, fontWeight: "900" }, candidateFooter: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginTop: 12 }, detailButton: { color: colors.accent, fontSize: 11, fontWeight: "900" }, rejectionBox: { backgroundColor: "#251F1D", borderRadius: 14, padding: 12, marginTop: 12 }, rejectionTitle: { color: colors.warning, fontWeight: "900", fontSize: 11, marginBottom: 5 }, rejectionText: { color: "#D7C7B0", fontSize: 11, lineHeight: 17 }, modelReason: { color: colors.muted, fontSize: 10, lineHeight: 15, marginTop: 8 },
  sportTabs: { gap: 9, paddingVertical: 6, paddingRight: 18, marginBottom: 16 }, sportTab: { borderWidth: 1, borderColor: colors.border, backgroundColor: colors.surface, borderRadius: 15, alignItems: "center", paddingVertical: 10, paddingHorizontal: 13, minWidth: 78 }, tabEmoji: { fontSize: 20 }, sportTabText: { color: colors.muted, fontSize: 11, fontWeight: "800", marginTop: 4 },
  sectionHeader: { marginVertical: 14 }, sectionTitle: { color: colors.text, fontWeight: "900", fontSize: 23 }, sectionSubtitle: { color: colors.muted, marginTop: 4 }, summaryGrid: { flexDirection: "row", gap: 8, marginBottom: 14 }, summary: { flex: 1, backgroundColor: colors.surface, borderWidth: 1, borderColor: colors.border, borderRadius: 16, padding: 13 }, summaryLabel: { color: colors.muted, fontSize: 10 }, summaryValue: { color: colors.text, fontSize: 17, fontWeight: "900", marginTop: 5 },
  performanceRow: { flexDirection: "row", alignItems: "center", backgroundColor: colors.surface, borderWidth: 1, borderColor: colors.border, borderRadius: 18, padding: 13, marginBottom: 9 }, performanceName: { flex: 1, marginLeft: 10 }, performanceTitle: { color: colors.text, fontWeight: "800", fontSize: 15 }, performanceSub: { color: colors.muted, fontSize: 11, marginTop: 3 }, performanceMetric: { alignItems: "flex-end", minWidth: 62, marginLeft: 8 }, performanceValue: { color: colors.text, fontWeight: "800" },
  statusRow: { flexDirection: "row", alignItems: "center", backgroundColor: colors.surface, borderBottomWidth: 1, borderColor: colors.border, padding: 16 }, statusDot: { width: 10, height: 10, borderRadius: 5, marginRight: 13 }, statusText: { flex: 1 }, statusLabel: { color: colors.text, fontWeight: "800" }, statusValue: { color: colors.muted, marginTop: 3, fontSize: 12 }, infoCard: { backgroundColor: "#102844", borderRadius: 20, padding: 18, marginTop: 16 }, infoTitle: { color: colors.accent, fontWeight: "900", fontSize: 16 }, infoText: { color: "#B5C9E6", lineHeight: 21, marginTop: 8 },
  nav: { flexDirection: "row", borderTopWidth: 1, borderColor: colors.border, backgroundColor: "#0B1727", paddingVertical: 9, paddingBottom: 12 }, navItem: { flex: 1, alignItems: "center" }, navIcon: { color: colors.muted, fontSize: 20, fontWeight: "900" }, navLabel: { color: colors.muted, fontSize: 10, fontWeight: "700", marginTop: 3 }, navActive: { color: colors.primary },
});
