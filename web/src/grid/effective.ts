import type { ChordEvent, EventOverride } from '../api/types'

export class OverlapError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'OverlapError'
  }
}

/**
 * The timeline a person sees: the immutable AI baseline with their personal overrides on top.
 * Same behaviour as `apply_overrides` in DE/src/analysis_contract/overrides.py; both are
 * checked against one golden file (contracts/fixtures/charts/*.effective.json).
 */
export function applyOverrides(
  aiEvents: readonly ChordEvent[],
  overrides: readonly EventOverride[],
): ChordEvent[] {
  const byId = new Map(aiEvents.map((event) => [event.event_id, event]))
  for (const override of overrides) {
    if (override.op === 'delete') {
      byId.delete(override.event_id)
    } else {
      if (!override.event) throw new Error(`${override.op} override ${override.event_id} has no event`)
      byId.set(override.event_id, { ...override.event, is_user_edited: true })
    }
  }
  const effective = [...byId.values()].sort((a, b) => a.start_ms - b.start_ms)
  for (let i = 1; i < effective.length; i++) {
    const prev = effective[i - 1]
    const cur = effective[i]
    if (cur.start_ms < prev.end_ms) {
      throw new OverlapError(`events ${prev.event_id} and ${cur.event_id} overlap`)
    }
  }
  return effective
}
