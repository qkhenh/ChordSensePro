import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { I18nContext, type I18nValue } from './context'
import { dictionaries, LANGS, type Lang } from './messages'

const STORAGE_KEY = 'chordsense.lang'

// localStorage can be missing or throw (private mode, blocked site data).
function readStored(): Lang {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    return LANGS.find((l) => l === stored) ?? 'vi'
  } catch {
    return 'vi'
  }
}

function store(lang: Lang) {
  try {
    localStorage.setItem(STORAGE_KEY, lang)
  } catch {
    /* the choice simply will not survive a reload */
  }
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(readStored)

  useEffect(() => {
    document.documentElement.lang = lang
  }, [lang])

  const value = useMemo<I18nValue>(
    () => ({
      lang,
      t: dictionaries[lang],
      setLang(next) {
        store(next)
        setLangState(next)
      },
    }),
    [lang],
  )
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}
