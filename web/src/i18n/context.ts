import { createContext, useContext } from 'react'
import { dictionaries, type Lang, type Messages } from './messages'

export interface I18nValue {
  lang: Lang
  t: Messages
  setLang(lang: Lang): void
}

// Defaults to Vietnamese so a component rendered without a provider (unit tests) still works.
export const I18nContext = createContext<I18nValue>({ lang: 'vi', t: dictionaries.vi, setLang: () => {} })

export const useI18n = () => useContext(I18nContext)
