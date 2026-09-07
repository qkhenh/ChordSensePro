import { useRef, useState, useEffect } from 'react'
import { motion, AnimatePresence, useScroll, useTransform } from 'motion/react'
import {
  ChevronDown, Globe, Mail, MapPin, Send, Star, Check, X,
  Menu, ChevronRight, Play, ArrowRight, Plus, Minus,
  Music, Mic, Headphones, Target, Activity, Zap, Brain, Sparkles, AudioWaveform, Layers, BookOpen,
} from 'lucide-react'

/* ─────────────────────────────────────────────────────────────────────────── */
/* Global styles / keyframes                                                   */
/* ─────────────────────────────────────────────────────────────────────────── */
const GlobalStyles = () => (
  <style>{`
    @keyframes shine {
      0%   { transform: translateX(-120%) skewX(-20deg); }
      100% { transform: translateX(350%)  skewX(-20deg); }
    }
    @keyframes textGlow {
      0%, 100% { text-shadow: 0 0 20px rgba(167,139,250,0.5), 0 0 60px rgba(6,182,212,0.2); }
      50%       { text-shadow: 0 0 40px rgba(167,139,250,0.9), 0 0 100px rgba(6,182,212,0.4); }
    }
    @keyframes float {
      0%, 100% { transform: translateY(0px); }
      50%       { transform: translateY(-10px); }
    }
    @keyframes pulse-ring {
      0%   { transform: scale(1);   opacity: 0.6; }
      100% { transform: scale(1.8); opacity: 0; }
    }
    @keyframes gradientShift {
      0%   { background-position: 0% 50%; }
      50%  { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }
    @keyframes waveBar {
      0%, 100% { transform: scaleY(0.3); }
      50%       { transform: scaleY(1); }
    }
    .logo-wrap:hover .logo-shine { animation: shine 0.7s ease forwards; }
    .text-glow { animation: textGlow 3s ease-in-out infinite; }
    .float-anim { animation: float 5s ease-in-out infinite; }
    .gradient-shift {
      background-size: 200% 200%;
      animation: gradientShift 4s ease infinite;
    }
    .wave-bar { animation: waveBar 1.2s ease-in-out infinite; }
    .wave-bar:nth-child(2) { animation-delay: 0.1s; }
    .wave-bar:nth-child(3) { animation-delay: 0.2s; }
    .wave-bar:nth-child(4) { animation-delay: 0.3s; }
    .wave-bar:nth-child(5) { animation-delay: 0.4s; }
    .wave-bar:nth-child(6) { animation-delay: 0.15s; }
    .wave-bar:nth-child(7) { animation-delay: 0.25s; }

    /* FAQ accordion */
    .faq-content {
      overflow: hidden;
      max-height: 0;
      transition: max-height 0.4s ease;
    }
    .faq-content.open {
      max-height: 300px;
    }

    /* Pricing card glow */
    .pricing-pro {
      box-shadow: 0 0 0 1px rgba(167,139,250,0.4), 0 0 60px rgba(124,58,237,0.25), 0 24px 64px rgba(0,0,0,0.5);
    }
    .pricing-pro:hover {
      box-shadow: 0 0 0 1px rgba(167,139,250,0.7), 0 0 80px rgba(124,58,237,0.35), 0 24px 64px rgba(0,0,0,0.5);
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.4); border-radius: 999px; }
  `}</style>
)

/* ─────────────────────────────────────────────────────────────────────────── */
/* Logo component                                                               */
/* ─────────────────────────────────────────────────────────────────────────── */
const LogoIcon = ({ size = 48 }) => (
  <div
    className="logo-wrap"
    style={{
      width: size, height: size, borderRadius: size * 0.22,
      position: 'relative', overflow: 'hidden', flexShrink: 0,
      boxShadow: `0 0 ${size * 0.8}px rgba(124,58,237,0.6),
                  0 0 ${size * 0.3}px rgba(6,182,212,0.35),
                  0 0 0 1px rgba(167,139,250,0.3)`,
    }}
  >
    <div className="logo-shine" style={{
      position: 'absolute', top: 0, left: 0, width: '45%', height: '100%',
      background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
      transform: 'translateX(-120%) skewX(-20deg)',
      pointerEvents: 'none', zIndex: 2,
    }} />
    <img
      src="/draft_logo2.jpg"
      alt="ChordSense Pro Logo"
      style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
    />
  </div>
)

/* ─────────────────────────────────────────────────────────────────────────── */
/* Navbar                                                                       */
/* ─────────────────────────────────────────────────────────────────────────── */
const Navbar = () => {
  const [scrolled, setScrolled] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const links = [
    { label: 'Features', href: '#features' },
    { label: 'How It Works', href: '#how-it-works' },
    { label: 'Pricing', href: '#pricing' },
    { label: 'FAQ', href: '#faq' },
    { label: 'Contact', href: '#contact' },
  ]

  const navLinkStyle = {
    color: 'rgba(255,255,255,0.7)', fontSize: 14, fontWeight: 500,
    textDecoration: 'none', transition: 'color 0.2s', padding: '4px 0',
  }

  return (
    <motion.nav
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 1000,
        padding: '0 24px',
        background: scrolled ? 'rgba(4,4,15,0.85)' : 'transparent',
        backdropFilter: scrolled ? 'blur(20px)' : 'none',
        borderBottom: scrolled ? '1px solid rgba(255,255,255,0.07)' : '1px solid transparent',
        transition: 'all 0.4s ease',
      }}
    >
      <div style={{
        maxWidth: 1200, margin: '0 auto',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        height: 68,
      }}>
        {/* Logo */}
        <a href="#" style={{ display: 'flex', alignItems: 'center', gap: 10, textDecoration: 'none' }}>
          <LogoIcon size={36} />
          <span style={{ fontFamily: "'Outfit', sans-serif", fontSize: 18, fontWeight: 700, color: '#fff', letterSpacing: '-0.02em' }}>
            ChordSense<span style={{ background: 'linear-gradient(90deg,#a78bfa,#67e8f9)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Pro</span>
          </span>
        </a>

        {/* Desktop links */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 32 }} className="desktop-nav">
          {links.map(l => (
            <a key={l.label} href={l.href} style={navLinkStyle}
              onMouseEnter={e => e.target.style.color = '#fff'}
              onMouseLeave={e => e.target.style.color = 'rgba(255,255,255,0.7)'}
            >{l.label}</a>
          ))}
        </div>

        {/* CTA */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <motion.a
            href="#pricing"
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.97 }}
            style={{
              padding: '9px 22px', borderRadius: 999,
              background: 'linear-gradient(135deg, #7C3AED, #06B6D4)',
              color: '#fff', fontSize: 13, fontWeight: 600,
              textDecoration: 'none', cursor: 'pointer',
              boxShadow: '0 0 20px rgba(124,58,237,0.4)',
            }}
          >
            Get Early Access
          </motion.a>
          <button
            style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.7)', cursor: 'pointer', display: 'none' }}
            className="mobile-menu-btn"
            onClick={() => setMobileOpen(!mobileOpen)}
          >
            <Menu size={22} />
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            style={{
              background: 'rgba(4,4,15,0.97)', backdropFilter: 'blur(20px)',
              borderTop: '1px solid rgba(255,255,255,0.07)',
              padding: '16px 24px 24px',
            }}
          >
            {links.map(l => (
              <a key={l.label} href={l.href}
                onClick={() => setMobileOpen(false)}
                style={{ ...navLinkStyle, display: 'block', padding: '12px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}
              >{l.label}</a>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      <style>{`
        @media (max-width: 768px) {
          .desktop-nav { display: none !important; }
          .mobile-menu-btn { display: block !important; }
        }
      `}</style>
    </motion.nav>
  )
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Animated waveform visual                                                     */
/* ─────────────────────────────────────────────────────────────────────────── */
const WaveformVisual = () => (
  <div style={{
    display: 'flex', alignItems: 'center', gap: 5,
    height: 64, padding: '0 8px',
  }}>
    {[0.4, 0.7, 1, 0.85, 0.55, 0.9, 0.65, 1, 0.75, 0.5, 0.8, 0.45].map((h, i) => (
      <div
        key={i}
        className="wave-bar"
        style={{
          width: 5, borderRadius: 999,
          height: `${h * 100}%`,
          background: i % 3 === 0
            ? 'linear-gradient(to top, #7C3AED, #a78bfa)'
            : i % 3 === 1
              ? 'linear-gradient(to top, #06B6D4, #67e8f9)'
              : 'linear-gradient(to top, #5b21b6, #7C3AED)',
          animationDelay: `${i * 0.08}s`,
        }}
      />
    ))}
  </div>
)

/* ─────────────────────────────────────────────────────────────────────────── */
/* Feature Pill (hero)                                                          */
/* ─────────────────────────────────────────────────────────────────────────── */
const FeaturePill = ({ icon: Icon, label, delay }) => (
  <motion.div
    initial={{ opacity: 0, y: 16 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay }}
    style={{
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '9px 20px', borderRadius: 999,
      background: 'rgba(255,255,255,0.06)',
      border: '1px solid rgba(255,255,255,0.14)',
      backdropFilter: 'blur(10px)',
      color: 'rgba(255,255,255,0.82)',
      fontSize: 13, fontWeight: 500, whiteSpace: 'nowrap',
      boxShadow: '0 1px 0 rgba(255,255,255,0.07) inset',
    }}
  >
    <Icon size={15} style={{ color: '#a78bfa' }} />
    {label}
  </motion.div>
)

/* ─────────────────────────────────────────────────────────────────────────── */
/* Glow orbs helper                                                             */
/* ─────────────────────────────────────────────────────────────────────────── */
const GlowOrbs = ({ orbs }) =>
  orbs.map((o, i) => (
    <div key={i} style={{
      position: 'absolute', width: o.w, height: o.h, borderRadius: '50%',
      top: o.top, bottom: o.bottom, left: o.left, right: o.right,
      background: `radial-gradient(circle, ${o.color} 0%, transparent 70%)`,
      filter: 'blur(60px)', pointerEvents: 'none',
    }} />
  ))

/* ─────────────────────────────────────────────────────────────────────────── */
/* Section heading helper                                                       */
/* ─────────────────────────────────────────────────────────────────────────── */
const SectionHeading = ({ badge, title, highlight, subtitle }) => (
  <div style={{ textAlign: 'center', marginBottom: 64 }}>
    {badge && (
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          padding: '6px 16px', borderRadius: 999, marginBottom: 20,
          background: 'rgba(124,58,237,0.15)', border: '1px solid rgba(124,58,237,0.35)',
          color: '#a78bfa', fontSize: 12, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase',
        }}
      >
        {badge}
      </motion.div>
    )}
    <motion.h2
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6, delay: 0.1 }}
      style={{
        fontFamily: "'Outfit', sans-serif",
        fontSize: 'clamp(28px, 4.5vw, 52px)', fontWeight: 800,
        letterSpacing: '-0.03em', color: '#fff',
        margin: '0 0 16px', lineHeight: 1.1,
      }}
    >
      {title}{' '}
      {highlight && (
        <span style={{
          background: 'linear-gradient(90deg, #c4b5fd, #67e8f9, #a78bfa)',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
        }}>
          {highlight}
        </span>
      )}
    </motion.h2>
    {subtitle && (
      <motion.p
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay: 0.2 }}
        style={{ color: 'rgba(255,255,255,0.45)', fontSize: 17, fontWeight: 400, maxWidth: 560, margin: '0 auto', lineHeight: 1.65 }}
      >
        {subtitle}
      </motion.p>
    )}
  </div>
)

/* ─────────────────────────────────────────────────────────────────────────── */
/* Animated counter                                                             */
/* ─────────────────────────────────────────────────────────────────────────── */
const Counter = ({ target, suffix = '', duration = 2000 }) => {
  const [count, setCount] = useState(0)
  const [started, setStarted] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !started) {
        setStarted(true)
        const start = Date.now()
        const tick = () => {
          const elapsed = Date.now() - start
          const progress = Math.min(elapsed / duration, 1)
          const eased = 1 - Math.pow(1 - progress, 3)
          setCount(Math.round(eased * target))
          if (progress < 1) requestAnimationFrame(tick)
        }
        requestAnimationFrame(tick)
      }
    }, { threshold: 0.5 })
    if (ref.current) observer.observe(ref.current)
    return () => observer.disconnect()
  }, [target, duration, started])

  return <span ref={ref}>{count.toLocaleString()}{suffix}</span>
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* LinkColumn (footer)                                                          */
/* ─────────────────────────────────────────────────────────────────────────── */
const LinkColumn = ({ title, links }) => (
  <div>
    <h4 style={{ textTransform: 'uppercase', letterSpacing: '0.14em', fontSize: 11, fontWeight: 700, color: '#111', marginBottom: 16, margin: '0 0 16px' }}>
      {title}
    </h4>
    <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>
      {links.map(link => (
        <li key={link}>
          <a href="#" style={{ color: '#6b7280', fontWeight: 500, fontSize: 14, textDecoration: 'none', transition: 'color 0.2s' }}
            onMouseEnter={e => e.target.style.color = '#7C3AED'}
            onMouseLeave={e => e.target.style.color = '#6b7280'}
          >{link}</a>
        </li>
      ))}
    </ul>
  </div>
)

/* ─────────────────────────────────────────────────────────────────────────── */
/* Music SVG Icons (outline style, like ChordAI)                               */
/* ─────────────────────────────────────────────────────────────────────────── */
const MusicIcons = {
  SmartGrid: ({ size = 48, color = '#67e8f9' }) => (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="6" y="6" width="36" height="36" rx="3"/>
      <line x1="6" y1="18" x2="42" y2="18"/>
      <line x1="6" y1="30" x2="42" y2="30"/>
      <line x1="18" y1="6" x2="18" y2="42"/>
      <line x1="30" y1="6" x2="30" y2="42"/>
      <rect x="6" y="18" width="12" height="12" fill={color} fillOpacity="0.2" stroke={color}/>
      <rect x="30" y="6" width="12" height="12" fill={color} fillOpacity="0.15" stroke={color}/>
    </svg>
  ),
  HarmonicEngine: ({ size = 48, color = '#a78bfa' }) => (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M26 7 C26 7 21 10 21 17 C21 23 26 23 26 30 C26 37 21 39 21 43"/>
      <path d="M21 21 C21 21 30 21 30 27 C30 33 21 33 21 27"/>
      <line x1="13" y1="24" x2="19" y2="24"/>
      <line x1="13" y1="29" x2="19" y2="29"/>
      <line x1="13" y1="34" x2="19" y2="34"/>
      <circle cx="26" cy="44" r="2.5" fill={color} stroke="none"/>
    </svg>
  ),
  ABLoop: ({ size = 48, color = '#86efac' }) => (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 24 A16 16 0 1 1 32 38"/>
      <polyline points="28,42 32,38 36,42"/>
      <text x="9" y="22" fontSize="10" fontWeight="700" fill={color} stroke="none" fontFamily="sans-serif">A</text>
      <text x="30" y="22" fontSize="10" fontWeight="700" fill={color} stroke="none" fontFamily="sans-serif">B</text>
      <line x1="13" y1="24" x2="13" y2="33"/>
      <line x1="34" y1="24" x2="34" y2="33"/>
    </svg>
  ),
  BeatTracking: ({ size = 48, color = '#fde68a' }) => (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="24,5 40,43 8,43"/>
      <line x1="24" y1="43" x2="24" y2="15"/>
      <line x1="24" y1="15" x2="34" y2="25" strokeWidth="2.2"/>
      <line x1="15" y1="28" x2="18" y2="28"/>
      <line x1="14" y1="34" x2="17" y2="34"/>
      <circle cx="24" cy="43" r="2" fill={color} stroke="none"/>
    </svg>
  ),
  ChordDictionary: ({ size = 48, color = '#fb923c' }) => (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="12" y="7" width="24" height="34" rx="2"/>
      <line x1="12" y1="17" x2="36" y2="17"/>
      <line x1="12" y1="27" x2="36" y2="27"/>
      <line x1="12" y1="37" x2="36" y2="37"/>
      <line x1="20" y1="7" x2="20" y2="41"/>
      <line x1="28" y1="7" x2="28" y2="41"/>
      <circle cx="20" cy="12" r="2.5" fill={color} stroke="none"/>
      <circle cx="28" cy="22" r="2.5" fill={color} stroke="none"/>
      <circle cx="20" cy="32" r="2.5" fill={color} stroke="none"/>
    </svg>
  ),
  SourceSeparation: ({ size = 48, color = '#f472b6' }) => (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none" stroke={color} strokeWidth="1.6" strokeLinecap="round">
      <line x1="4" y1="10" x2="44" y2="10"/>
      <rect x="12" y="7" width="8" height="6" rx="1" fill={color} fillOpacity="0.3"/>
      <rect x="24" y="7" width="12" height="6" rx="1" fill={color} fillOpacity="0.2"/>
      <line x1="4" y1="20" x2="44" y2="20"/>
      <rect x="8" y="17" width="14" height="6" rx="1" fill={color} fillOpacity="0.3"/>
      <rect x="26" y="17" width="8" height="6" rx="1" fill={color} fillOpacity="0.2"/>
      <line x1="4" y1="30" x2="44" y2="30"/>
      <rect x="16" y="27" width="10" height="6" rx="1" fill={color} fillOpacity="0.3"/>
      <rect x="30" y="27" width="6" height="6" rx="1" fill={color} fillOpacity="0.2"/>
      <line x1="4" y1="40" x2="44" y2="40"/>
      <rect x="10" y="37" width="16" height="6" rx="1" fill={color} fillOpacity="0.3"/>
      <rect x="28" y="37" width="10" height="6" rx="1" fill={color} fillOpacity="0.2"/>
    </svg>
  ),
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* APP                                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */
export default function App() {
  const [openFaq, setOpenFaq] = useState(null)
  const [activeTestimonial, setActiveTestimonial] = useState(0)
  const [email, setEmail] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [contactForm, setContactForm] = useState({ name: '', email: '', message: '' })
  const [contactSent, setContactSent] = useState(false)

  // Auto-rotate testimonials
  useEffect(() => {
    const t = setInterval(() => setActiveTestimonial(p => (p + 1) % 3), 5000)
    return () => clearInterval(t)
  }, [])

  /* ─── DATA ─── */
  const features = [
    {
      SvgIcon: MusicIcons.SmartGrid,
      color: '#67e8f9',
      glow: 'rgba(103,232,249,0.12)',
      title: 'Smart Grid View',
      desc: 'Lưới hợp âm đồng bộ với audio — click vào ô nhịp để tua nhạc đến đúng mili-giây (<100ms). Xem toàn bộ chord progression bài nhạc trực quan.',
    },
    {
      SvgIcon: MusicIcons.HarmonicEngine,
      color: '#a78bfa',
      glow: 'rgba(167,139,250,0.15)',
      title: 'Harmonic Analysis Engine',
      desc: 'Tự động phân tích bậc hòa âm (Roman numerals: I, ii, V7…), gợi ý âm giai (Ionian, Mixolydian…) — không có trên Chordify hay ChordAI.',
    },
    {
      SvgIcon: MusicIcons.ABLoop,
      color: '#86efac',
      glow: 'rgba(134,239,172,0.12)',
      title: 'A/B Loop & Practice Tools',
      desc: 'Quét chọn đoạn khó → lặp lại A/B Loop liên tục. Metronome động theo BPM bài. Speed shifting 0.5×–1× không méo cao độ.',
    },
    {
      SvgIcon: MusicIcons.BeatTracking,
      color: '#fde68a',
      glow: 'rgba(253,230,138,0.12)',
      title: 'Beat & Key Tracking',
      desc: 'madmom Beat Tracker phát hiện tempo, downbeat, và time signature. Key Detection tự động chọn tên nốt đúng (C# vs Db) theo giọng bài.',
    },
    {
      SvgIcon: MusicIcons.ChordDictionary,
      color: '#fb923c',
      glow: 'rgba(251,146,60,0.12)',
      title: 'Interactive Chord Dictionary',
      desc: 'Click vào hợp âm bất kỳ → mở modal tra cứu fretboard (guitar) và phím (piano). Phát âm thanh qua Soundfont Synth ngay trên browser.',
    },
    {
      SvgIcon: MusicIcons.SourceSeparation,
      color: '#f472b6',
      glow: 'rgba(244,114,182,0.12)',
      title: 'AI Source Separation',
      desc: 'Demucs (Meta AI) tách stems: drums / bass / vocals / other — chỉ phân tích track piano/guitar sạch, tăng accuracy extended chord detection.',
    },
  ]

  const stats = [
    { value: 1200, suffix: '+', label: 'Chord Variations', icon: Music },
    { value: 94.7, suffix: '%', label: 'Recognition Accuracy', icon: Target },
    { value: 50000, suffix: '+', label: 'Training Audio Samples', icon: Activity },
    { value: 500, suffix: '+', label: 'Beta Musicians', icon: Headphones },
  ]

  const steps = [
    {
      num: '01', icon: Music, color: '#a78bfa',
      title: 'Import Your Song',
      desc: 'Paste YouTube URL hoặc upload MP3/WAV. Backend dùng yt-dlp tải audio → Demucs tách stems → pipeline AI xử lý (~10-30 giây).',
    },
    {
      num: '02', icon: Brain, color: '#67e8f9',
      title: 'AI Phân Tích Hòa Âm',
      desc: 'MERT-v1-330M + LoRA Hierarchical nhận diện extended chords (7th, 9th, 11th, 13th). Harmonic Engine tự động map sang Roman numerals và âm giai.',
    },
    {
      num: '03', icon: Target, color: '#86efac',
      title: 'Luyện Tập & Thành Thạo',
      desc: 'Smart Grid đồng bộ audio — A/B Loop đoạn khó, Metronome, Speed Shifting. Chord Dictionary tra cứu ngay. Lưu playlist cá nhân.',
    },
  ]

  const testimonials = [
    {
      name: 'Nguyễn Minh Trí',
      role: 'Jazz Pianist, HCM Conservatory',
      avatar: 'NMT',
      color: '#a78bfa',
      stars: 5,
      text: '"ChordSense Pro identified altered dominants in my Herbie Hancock transcriptions that I\'d been mis-hearing for years. The harmonic analysis is genuinely impressive — this is the tool I\'ve been waiting for."',
    },
    {
      name: 'Lê Thị Bích Hà',
      role: 'Music Educator, HCMUT',
      avatar: 'LBH',
      color: '#67e8f9',
      stars: 5,
      text: '"I use ChordSense Pro in my music theory classes. Students finally understand chord functions visually. The Smart Grid View makes abstract concepts tangible. Highly recommended for educators."',
    },
    {
      name: 'Trần Hoàng Phúc',
      role: 'Songwriter & Producer',
      avatar: 'THP',
      color: '#86efac',
      stars: 5,
      text: '"Real-time chord detection while I improvise has transformed my songwriting process. I catch interesting progressions I\'d normally forget. It\'s like having a music theory expert in the room."',
    },
  ]

  const plans = [
    {
      name: 'Student',
      price: 'Free',
      sub: 'Forever free for learners',
      color: '#6b7280',
      features: [
        'Basic chord recognition (triads & 7ths)',
        'Real-time analysis (5 min sessions)',
        'Chord dictionary access',
        '3 practice sessions / day',
        'Community support',
      ],
      missing: ['Extended chord detection', 'Smart Grid View', 'Harmonic analysis reports', 'Unlimited sessions'],
      cta: 'Start Free',
      pro: false,
    },
    {
      name: 'Pro',
      price: '₫199k',
      sub: 'per month • cancel anytime',
      color: '#a78bfa',
      features: [
        'Full extended chord recognition',
        'Unlimited real-time sessions',
        'Smart Grid View & harmonic analysis',
        'Vietnamese Piano Dataset tuning',
        'Practice workspace & progress tracking',
        'AI-generated practice recommendations',
        'Priority support & early features',
        'Export chord charts as PDF',
      ],
      missing: [],
      cta: 'Get Early Access',
      pro: true,
    },
  ]

  const faqs = [
    {
      q: 'ChordSense Pro hoạt động với nhạc cụ nào?',
      a: 'Bất kỳ nhạc cụ nào có thể ghi âm qua microphone hoặc xuất file audio đều được hỗ trợ — piano, guitar, bass, organ, và cả giọng hát. Chúng tôi đặc biệt tối ưu cho piano với Vietnamese Piano Dataset.',
    },
    {
      q: 'Độ chính xác nhận diện hợp âm là bao nhiêu?',
      a: 'Trong benchmark nội bộ với Vietnamese Piano Dataset, ChordSense Pro đạt 94.7% accuracy trên extended chords. Với basic triads, accuracy vượt 98%. Kết quả tốt nhất với piano trong điều kiện không có nhiễu.',
    },
    {
      q: 'AI model được train trên dữ liệu nào?',
      a: 'Chúng tôi fine-tune MERT-v1-330M (pretrain trên 160k+ giờ nhạc) với Vietnamese Piano Dataset được thu thập đặc biệt tại Việt Nam, bổ sung thêm các bộ dữ liệu nhạc cổ điển và jazz quốc tế.',
    },
    {
      q: 'Phiên bản Free có giới hạn gì không?',
      a: 'Phiên bản Free giới hạn ở basic chords (triads & 7ths), sessions 5 phút, và 3 sessions/ngày. Để nhận diện extended chords (9th, 11th, 13th, altered), Smart Grid View, và không giới hạn session, bạn cần nâng cấp lên Pro.',
    },
    {
      q: 'Có cần kết nối internet không?',
      a: 'Hiện tại ChordSense Pro yêu cầu kết nối internet để xử lý AI inference. Chúng tôi đang phát triển phiên bản offline cho các phiên bản tương lai.',
    },
    {
      q: 'Có thể hủy đăng ký bất kỳ lúc nào không?',
      a: 'Có, hoàn toàn. Không có hợp đồng dài hạn hay phí hủy. Bạn có thể hủy bất kỳ lúc nào và vẫn giữ quyền truy cập Pro đến hết chu kỳ thanh toán.',
    },
  ]

  /* ─── RENDER ─── */
  return (
    <div style={{ fontFamily: "'Inter', sans-serif", background: '#04040f', minHeight: '100vh' }}>
      <GlobalStyles />
      <Navbar />

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 1. HERO SECTION                                                    */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section style={{
        minHeight: '100vh',
        background: 'linear-gradient(160deg, #04040f 0%, #0b0320 50%, #060d20 100%)',
        position: 'relative', overflow: 'hidden',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        padding: '120px 24px 80px',
      }}>
        <GlowOrbs orbs={[
          { top: '10%', left: '15%', w: 700, h: 700, color: 'rgba(109,40,217,0.2)' },
          { top: '55%', right: '8%',  w: 500, h: 500, color: 'rgba(6,182,212,0.14)' },
          { bottom: '5%', left: '35%', w: 350, h: 350, color: 'rgba(167,139,250,0.1)' },
        ]} />

        {/* Grid */}
        <div style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          backgroundImage: 'linear-gradient(rgba(255,255,255,0.022) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.022) 1px, transparent 1px)',
          backgroundSize: '64px 64px',
        }} />
        {/* Vignette */}
        <div style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          background: 'radial-gradient(ellipse at center, transparent 30%, rgba(4,4,15,0.75) 100%)',
        }} />

        <div style={{ position: 'relative', textAlign: 'center', maxWidth: 820, zIndex: 1 }}>

          {/* Logo */}
          <motion.div
            initial={{ opacity: 0, scale: 0.75 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7, ease: [0.34, 1.56, 0.64, 1] }}
            style={{ display: 'flex', justifyContent: 'center', marginBottom: 32 }}
            className="float-anim"
          >
            <LogoIcon size={88} />
          </motion.div>

          {/* Title */}
          <motion.h1
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.18 }}
            className="text-glow"
            style={{
              fontFamily: "'Outfit', sans-serif",
              fontSize: 'clamp(44px, 8vw, 88px)',
              fontWeight: 900, letterSpacing: '-0.04em',
              color: '#fff', margin: '0 0 20px', lineHeight: 1.0,
            }}
          >
            ChordSense
            <span style={{
              background: 'linear-gradient(90deg, #c4b5fd, #67e8f9, #a78bfa)',
              backgroundSize: '200% auto',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
              filter: 'drop-shadow(0 0 16px rgba(167,139,250,0.7))',
            }}> Pro</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            style={{
              fontSize: 'clamp(15px, 2vw, 19px)',
              color: 'rgba(255,255,255,0.5)', fontWeight: 400,
              margin: '0 0 24px', lineHeight: 1.7, maxWidth: 620, marginLeft: 'auto', marginRight: 'auto',
            }}
          >
            An AI-Powered Context-Aware System for Musicians —<br />
            Real-time chord detection, extended harmonic analysis,<br />
            and intelligent practice recommendations.
          </motion.p>


          {/* CTA buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
            style={{ display: 'flex', gap: 16, justifyContent: 'center', flexWrap: 'wrap', marginBottom: 64 }}
          >
            <motion.a
              href="#pricing"
              whileHover={{ scale: 1.05, boxShadow: '0 0 40px rgba(124,58,237,0.6)' }}
              whileTap={{ scale: 0.97 }}
              style={{
                padding: '14px 36px', borderRadius: 999, cursor: 'pointer',
                background: 'linear-gradient(135deg, #7C3AED 0%, #06B6D4 100%)',
                color: '#fff', fontSize: 15, fontWeight: 700,
                textDecoration: 'none',
                boxShadow: '0 0 30px rgba(124,58,237,0.45)',
                display: 'flex', alignItems: 'center', gap: 8,
              }}
            >
              Get Early Access <ArrowRight size={16} />
            </motion.a>
            <motion.a
              href="#how-it-works"
              whileHover={{ scale: 1.04, background: 'rgba(255,255,255,0.1)' }}
              whileTap={{ scale: 0.97 }}
              style={{
                padding: '14px 36px', borderRadius: 999, cursor: 'pointer',
                background: 'rgba(255,255,255,0.07)',
                border: '1px solid rgba(255,255,255,0.18)',
                color: '#fff', fontSize: 15, fontWeight: 600,
                textDecoration: 'none',
                display: 'flex', alignItems: 'center', gap: 8,
                backdropFilter: 'blur(8px)',
              }}
            >
              <Play size={16} style={{ fill: '#fff' }} /> See How It Works
            </motion.a>
          </motion.div>

          {/* Feature pills */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, justifyContent: 'center', marginBottom: 64 }}>
            <FeaturePill icon={Brain}         label="MERT-v1-330M AI"           delay={0.62} />
            <FeaturePill icon={AudioWaveform} label="Extended Chord Recognition" delay={0.70} />
            <FeaturePill icon={Zap}           label="Real-time Practice"         delay={0.78} />
            <FeaturePill icon={Sparkles}      label="Harmonic Analysis"          delay={0.86} />
          </div>

          {/* Scroll indicator */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.2 }}
            onClick={() => window.scrollTo({ top: window.innerHeight, behavior: 'smooth' })}
            style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, color: 'rgba(255,255,255,0.25)', cursor: 'pointer' }}
          >
            <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.22em', textTransform: 'uppercase' }}>
              Scroll to explore
            </span>
            <motion.div animate={{ y: [0, 7, 0] }} transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}>
              <ChevronDown size={20} />
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 2. STATS SECTION                                                   */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section style={{
        background: 'linear-gradient(180deg, #04040f 0%, #07021a 100%)',
        padding: '80px 24px',
        position: 'relative', overflow: 'hidden',
        borderTop: '1px solid rgba(255,255,255,0.05)',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
      }}>
        {/* Subtle center glow */}
        <div style={{
          position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)',
          width: 800, height: 300, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(124,58,237,0.1) 0%, transparent 70%)',
          filter: 'blur(40px)', pointerEvents: 'none',
        }} />

        <div style={{ maxWidth: 1100, margin: '0 auto', position: 'relative', zIndex: 1 }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: 2,
          }}>
            {stats.map(({ value, suffix, label, icon: Icon }, i) => (
              <motion.div
                key={label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                style={{
                  textAlign: 'center', padding: '40px 24px',
                  borderRight: i < stats.length - 1 ? '1px solid rgba(255,255,255,0.07)' : 'none',
                }}
              >
                <Icon size={22} style={{ color: '#7C3AED', marginBottom: 12 }} />
                <div style={{
                  fontFamily: "'Outfit', sans-serif",
                  fontSize: 'clamp(36px, 5vw, 52px)', fontWeight: 800,
                  color: '#fff', letterSpacing: '-0.03em', lineHeight: 1,
                  marginBottom: 8,
                  background: 'linear-gradient(135deg, #c4b5fd, #67e8f9)',
                  WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                }}>
                  <Counter target={value} suffix={suffix} />
                </div>
                <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: 13, fontWeight: 500 }}>{label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 3. FEATURES SECTION                                                */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section id="features" style={{
        background: '#07021a',
        padding: '120px 24px',
        position: 'relative', overflow: 'hidden',
      }}>
        <GlowOrbs orbs={[
          { top: '0%', right: '10%', w: 500, h: 500, color: 'rgba(6,182,212,0.1)' },
          { bottom: '0%', left: '5%', w: 400, h: 400, color: 'rgba(124,58,237,0.12)' },
        ]} />

        <div style={{ maxWidth: 1200, margin: '0 auto', position: 'relative', zIndex: 1 }}>
          <SectionHeading
            badge="✦ Features"
            title="Everything You Need to"
            highlight="Master Harmony"
            subtitle="Professional-grade chord intelligence built for musicians who want to go deeper than surface-level analysis."
          />

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
            gap: 20,
          }}>
            {features.map(({ SvgIcon, color, glow, title, desc }, i) => (
              <motion.div
                key={title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.08 }}
                whileHover={{ y: -4, boxShadow: `0 24px 64px rgba(0,0,0,0.5), 0 0 0 1px ${color}30` }}
                style={{
                  background: 'rgba(255,255,255,0.04)',
                  border: '1px solid rgba(255,255,255,0.08)',
                  borderRadius: 20, padding: '32px 28px',
                  backdropFilter: 'blur(10px)',
                  cursor: 'default',
                  transition: 'all 0.3s ease',
                }}
              >
                <div style={{
                  width: 72, height: 72, borderRadius: 16, marginBottom: 20,
                  background: glow, border: `1px solid ${color}30`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <SvgIcon size={40} color={color} />
                </div>
                <h3 style={{
                  fontFamily: "'Outfit', sans-serif",
                  color: '#fff', fontWeight: 700, fontSize: 18,
                  margin: '0 0 12px', letterSpacing: '-0.01em',
                }}>{title}</h3>
                <p style={{ color: 'rgba(255,255,255,0.45)', fontSize: 14, lineHeight: 1.7, margin: 0 }}>{desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 4. HOW IT WORKS                                                    */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section id="how-it-works" style={{
        background: 'linear-gradient(180deg, #07021a 0%, #04040f 100%)',
        padding: '120px 24px',
        position: 'relative', overflow: 'hidden',
      }}>
        <div style={{ maxWidth: 1100, margin: '0 auto', position: 'relative', zIndex: 1 }}>
          <SectionHeading
            badge="⚡ How It Works"
            title="From Audio to"
            highlight="Insight in Seconds"
            subtitle="Three simple steps to transform how you understand and practice music."
          />

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: 40,
          }}>
            {steps.map(({ num, icon: Icon, color, title, desc }, i) => (
              <motion.div
                key={num}
                initial={{ opacity: 0, x: i % 2 === 0 ? -24 : 24 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: i * 0.15 }}
                style={{ position: 'relative', textAlign: 'center' }}
              >
                {/* Connector line */}
                {i < steps.length - 1 && (
                  <div style={{
                    position: 'absolute', top: 36, right: '-20px',
                    width: 40, height: 2,
                    background: `linear-gradient(90deg, ${color}, rgba(255,255,255,0.1))`,
                    display: 'none', // Hidden on mobile, shown on wider screens via media
                  }} />
                )}

                {/* Step number + icon */}
                <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 24 }}>
                  <div style={{ position: 'relative' }}>
                    <div style={{
                      width: 72, height: 72, borderRadius: '50%',
                      background: `${color}15`,
                      border: `2px solid ${color}40`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      boxShadow: `0 0 40px ${color}20`,
                    }}>
                      <Icon size={28} style={{ color }} />
                    </div>
                    <div style={{
                      position: 'absolute', top: -6, right: -6,
                      width: 22, height: 22, borderRadius: '50%',
                      background: `linear-gradient(135deg, #7C3AED, #06B6D4)`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 10, fontWeight: 800, color: '#fff',
                    }}>
                      {i + 1}
                    </div>
                  </div>
                </div>

                <div style={{
                  padding: '3px 12px', borderRadius: 999,
                  background: `${color}18`, border: `1px solid ${color}35`,
                  color, fontSize: 11, fontWeight: 700, letterSpacing: '0.08em',
                  textTransform: 'uppercase', display: 'inline-block', marginBottom: 14,
                }}>
                  Step {num}
                </div>
                <h3 style={{
                  fontFamily: "'Outfit', sans-serif",
                  color: '#fff', fontSize: 22, fontWeight: 700,
                  margin: '0 0 12px', letterSpacing: '-0.02em',
                }}>{title}</h3>
                <p style={{ color: 'rgba(255,255,255,0.45)', fontSize: 14, lineHeight: 1.75, margin: 0 }}>{desc}</p>
              </motion.div>
            ))}
          </div>

          {/* Waveform visual */}
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.4 }}
            style={{
              marginTop: 80, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 24,
              padding: '32px 40px',
              background: 'rgba(124,58,237,0.08)',
              border: '1px solid rgba(124,58,237,0.2)',
              borderRadius: 24, backdropFilter: 'blur(10px)',
              maxWidth: 600, margin: '80px auto 0',
            }}
          >
            <div>
              <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12, fontWeight: 600, marginBottom: 4 }}>LIVE DETECTION</div>
              <div style={{ color: '#fff', fontSize: 18, fontWeight: 700 }}>Cmaj9 → Am7 → Fmaj7 → G13</div>
            </div>
            <WaveformVisual />
          </motion.div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 5. TESTIMONIALS                                                    */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section style={{
        background: '#07021a',
        padding: '120px 24px',
        position: 'relative', overflow: 'hidden',
        borderTop: '1px solid rgba(255,255,255,0.05)',
      }}>
        <GlowOrbs orbs={[
          { top: '30%', left: '5%', w: 400, h: 400, color: 'rgba(124,58,237,0.1)' },
          { top: '20%', right: '5%', w: 350, h: 350, color: 'rgba(6,182,212,0.08)' },
        ]} />

        <div style={{ maxWidth: 900, margin: '0 auto', position: 'relative', zIndex: 1 }}>
          <SectionHeading
            badge="💬 Testimonials"
            title="Loved by"
            highlight="Musicians"
            subtitle="Join hundreds of musicians already transforming their practice."
          />

          {/* Testimonial card */}
          <div style={{ position: 'relative', minHeight: 260 }}>
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTestimonial}
                initial={{ opacity: 0, x: 40 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -40 }}
                transition={{ duration: 0.4 }}
                style={{
                  background: 'rgba(255,255,255,0.04)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: 24, padding: '40px 48px',
                  backdropFilter: 'blur(10px)',
                  textAlign: 'center',
                }}
              >
                {/* Avatar */}
                <div style={{
                  width: 56, height: 56, borderRadius: '50%',
                  background: `linear-gradient(135deg, ${testimonials[activeTestimonial].color}, #06B6D4)`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 16, fontWeight: 700, color: '#fff',
                  margin: '0 auto 16px',
                  boxShadow: `0 0 24px ${testimonials[activeTestimonial].color}50`,
                }}>
                  {testimonials[activeTestimonial].avatar}
                </div>

                {/* Stars */}
                <div style={{ display: 'flex', justifyContent: 'center', gap: 4, marginBottom: 20 }}>
                  {[...Array(testimonials[activeTestimonial].stars)].map((_, i) => (
                    <Star key={i} size={16} style={{ color: '#fde68a', fill: '#fde68a' }} />
                  ))}
                </div>

                <p style={{
                  color: 'rgba(255,255,255,0.75)', fontSize: 16, lineHeight: 1.8,
                  fontStyle: 'italic', margin: '0 0 28px', maxWidth: 640, marginLeft: 'auto', marginRight: 'auto',
                }}>
                  {testimonials[activeTestimonial].text}
                </p>

                <div>
                  <div style={{ color: '#fff', fontWeight: 700, fontSize: 15 }}>
                    {testimonials[activeTestimonial].name}
                  </div>
                  <div style={{ color: 'rgba(255,255,255,0.35)', fontSize: 13, marginTop: 4 }}>
                    {testimonials[activeTestimonial].role}
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Dots */}
          <div style={{ display: 'flex', justifyContent: 'center', gap: 10, marginTop: 28 }}>
            {testimonials.map((_, i) => (
              <button
                key={i}
                onClick={() => setActiveTestimonial(i)}
                style={{
                  width: i === activeTestimonial ? 24 : 8,
                  height: 8, borderRadius: 999, border: 'none', cursor: 'pointer',
                  background: i === activeTestimonial ? '#a78bfa' : 'rgba(255,255,255,0.2)',
                  transition: 'all 0.3s ease',
                  padding: 0,
                }}
              />
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 6. PRICING                                                         */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section id="pricing" style={{
        background: 'linear-gradient(180deg, #07021a 0%, #04040f 100%)',
        padding: '120px 24px',
        position: 'relative', overflow: 'hidden',
      }}>
        <GlowOrbs orbs={[
          { top: '20%', left: '50%', w: 800, h: 400, color: 'rgba(124,58,237,0.08)' },
        ]} />

        <div style={{ maxWidth: 900, margin: '0 auto', position: 'relative', zIndex: 1 }}>
          <SectionHeading
            badge="💰 Pricing"
            title="Simple,"
            highlight="Transparent Pricing"
            subtitle="Start free, upgrade when you're ready. No hidden fees, no surprises."
          />

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
            gap: 24,
            alignItems: 'start',
          }}>
            {plans.map((plan, i) => (
              <motion.div
                key={plan.name}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.15 }}
                className={plan.pro ? 'pricing-pro' : ''}
                style={{
                  background: plan.pro
                    ? 'linear-gradient(160deg, rgba(124,58,237,0.18) 0%, rgba(6,182,212,0.08) 100%)'
                    : 'rgba(255,255,255,0.04)',
                  border: plan.pro
                    ? '1px solid rgba(167,139,250,0.3)'
                    : '1px solid rgba(255,255,255,0.08)',
                  borderRadius: 24,
                  overflow: 'hidden',
                  position: 'relative',
                  transition: 'all 0.3s ease',
                }}
              >
                {plan.pro && (
                  <div style={{
                    position: 'absolute', top: 20, right: 20,
                    padding: '4px 12px', borderRadius: 999,
                    background: 'linear-gradient(135deg, #7C3AED, #06B6D4)',
                    color: '#fff', fontSize: 11, fontWeight: 700, letterSpacing: '0.05em',
                  }}>
                    RECOMMENDED
                  </div>
                )}

                <div style={{ padding: '36px 32px 28px' }}>
                  <div style={{ color: plan.pro ? '#a78bfa' : '#6b7280', fontSize: 13, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 12 }}>
                    {plan.name}
                  </div>
                  <div style={{
                    fontFamily: "'Outfit', sans-serif",
                    fontSize: 'clamp(36px, 5vw, 48px)', fontWeight: 800,
                    color: '#fff', letterSpacing: '-0.03em', lineHeight: 1,
                    marginBottom: 6,
                  }}>
                    {plan.price}
                  </div>
                  <div style={{ color: 'rgba(255,255,255,0.35)', fontSize: 13, marginBottom: 32 }}>
                    {plan.sub}
                  </div>

                  <motion.a
                    href="#contact"
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    style={{
                      display: 'block', textAlign: 'center',
                      padding: '13px 24px', borderRadius: 999,
                      background: plan.pro ? 'linear-gradient(135deg, #7C3AED, #06B6D4)' : 'rgba(255,255,255,0.08)',
                      border: plan.pro ? 'none' : '1px solid rgba(255,255,255,0.15)',
                      color: '#fff', fontSize: 14, fontWeight: 700,
                      textDecoration: 'none', cursor: 'pointer',
                      boxShadow: plan.pro ? '0 0 24px rgba(124,58,237,0.4)' : 'none',
                      marginBottom: 32,
                    }}
                  >
                    {plan.cta}
                  </motion.a>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {plan.features.map(f => (
                      <div key={f} style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                        <Check size={15} style={{ color: plan.pro ? '#86efac' : '#6b7280', marginTop: 2, flexShrink: 0 }} />
                        <span style={{ color: 'rgba(255,255,255,0.7)', fontSize: 14 }}>{f}</span>
                      </div>
                    ))}
                    {plan.missing.map(f => (
                      <div key={f} style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                        <X size={15} style={{ color: 'rgba(255,255,255,0.2)', marginTop: 2, flexShrink: 0 }} />
                        <span style={{ color: 'rgba(255,255,255,0.3)', fontSize: 14 }}>{f}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 7. FAQ                                                             */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section id="faq" style={{
        background: '#07021a',
        padding: '120px 24px',
        borderTop: '1px solid rgba(255,255,255,0.05)',
      }}>
        <div style={{ maxWidth: 740, margin: '0 auto' }}>
          <SectionHeading
            badge="❓ FAQ"
            title="Common"
            highlight="Questions"
            subtitle="Everything you need to know before getting started."
          />

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {faqs.map((faq, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.06 }}
                style={{
                  background: 'rgba(255,255,255,0.04)',
                  border: openFaq === i ? '1px solid rgba(167,139,250,0.3)' : '1px solid rgba(255,255,255,0.07)',
                  borderRadius: 16, overflow: 'hidden',
                  transition: 'border-color 0.3s ease',
                }}
              >
                <button
                  onClick={() => setOpenFaq(openFaq === i ? null : i)}
                  style={{
                    width: '100%', padding: '20px 24px',
                    background: 'none', border: 'none', cursor: 'pointer',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16,
                    textAlign: 'left',
                  }}
                >
                  <span style={{ color: '#fff', fontSize: 15, fontWeight: 600, lineHeight: 1.4 }}>
                    {faq.q}
                  </span>
                  <span style={{
                    flexShrink: 0,
                    width: 28, height: 28, borderRadius: '50%',
                    background: openFaq === i ? 'rgba(124,58,237,0.3)' : 'rgba(255,255,255,0.07)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    transition: 'all 0.3s ease',
                  }}>
                    {openFaq === i
                      ? <Minus size={14} style={{ color: '#a78bfa' }} />
                      : <Plus size={14} style={{ color: 'rgba(255,255,255,0.5)' }} />
                    }
                  </span>
                </button>
                <AnimatePresence>
                  {openFaq === i && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.35, ease: 'easeInOut' }}
                      style={{ overflow: 'hidden' }}
                    >
                      <p style={{
                        color: 'rgba(255,255,255,0.5)', fontSize: 14, lineHeight: 1.75,
                        padding: '0 24px 20px', margin: 0,
                      }}>
                        {faq.a}
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 8. FINAL CTA BANNER                                                */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section style={{
        padding: '100px 24px',
        background: 'linear-gradient(160deg, #04040f 0%, #0b0320 60%, #060d20 100%)',
        position: 'relative', overflow: 'hidden',
        borderTop: '1px solid rgba(255,255,255,0.05)',
      }}>
        <GlowOrbs orbs={[
          { top: '50%', left: '50%', w: 900, h: 500, color: 'rgba(124,58,237,0.13)' },
        ]} />

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          style={{
            maxWidth: 680, margin: '0 auto', textAlign: 'center',
            position: 'relative', zIndex: 1,
          }}
        >
          <div style={{
            width: 64, height: 64, borderRadius: '50%',
            background: 'rgba(124,58,237,0.2)', border: '1px solid rgba(167,139,250,0.3)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 28px',
            boxShadow: '0 0 40px rgba(124,58,237,0.3)',
          }}>
            <Music size={28} style={{ color: '#a78bfa' }} />
          </div>

          <h2 style={{
            fontFamily: "'Outfit', sans-serif",
            fontSize: 'clamp(30px, 5vw, 52px)', fontWeight: 900,
            letterSpacing: '-0.03em', color: '#fff',
            margin: '0 0 16px', lineHeight: 1.1,
          }}>
            Start Your{' '}
            <span style={{
              background: 'linear-gradient(90deg, #c4b5fd, #67e8f9)',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
            }}>Musical Journey</span>{' '}
            Today
          </h2>

          <p style={{ color: 'rgba(255,255,255,0.45)', fontSize: 16, lineHeight: 1.7, margin: '0 0 40px' }}>
            Join the waitlist and be among the first to experience
            professional-grade chord intelligence.
          </p>

          {/* Email form */}
          {!submitted ? (
            <form
              onSubmit={e => { e.preventDefault(); if (email) setSubmitted(true) }}
              style={{
                display: 'flex', gap: 12, flexWrap: 'wrap', justifyContent: 'center',
                maxWidth: 480, margin: '0 auto 24px',
              }}
            >
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="your@email.com"
                required
                style={{
                  flex: 1, minWidth: 220,
                  padding: '13px 20px', borderRadius: 999,
                  background: 'rgba(255,255,255,0.07)',
                  border: '1px solid rgba(255,255,255,0.15)',
                  color: '#fff', fontSize: 14, outline: 'none',
                  fontFamily: "'Inter', sans-serif",
                }}
              />
              <motion.button
                type="submit"
                whileHover={{ scale: 1.05, boxShadow: '0 0 30px rgba(124,58,237,0.6)' }}
                whileTap={{ scale: 0.97 }}
                style={{
                  padding: '13px 28px', borderRadius: 999,
                  background: 'linear-gradient(135deg, #7C3AED, #06B6D4)',
                  color: '#fff', fontSize: 14, fontWeight: 700,
                  border: 'none', cursor: 'pointer',
                  display: 'flex', alignItems: 'center', gap: 8,
                  boxShadow: '0 0 20px rgba(124,58,237,0.4)',
                }}
              >
                <Send size={15} /> Notify Me
              </motion.button>
            </form>
          ) : (
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              style={{
                padding: '20px 32px', borderRadius: 16,
                background: 'rgba(134,239,172,0.12)',
                border: '1px solid rgba(134,239,172,0.3)',
                color: '#86efac', fontSize: 15, fontWeight: 600,
                maxWidth: 400, margin: '0 auto 24px',
                display: 'flex', alignItems: 'center', gap: 10, justifyContent: 'center',
              }}
            >
              <Check size={18} /> You're on the list! We'll reach out soon.
            </motion.div>
          )}

          <p style={{ color: 'rgba(255,255,255,0.2)', fontSize: 12 }}>
            No spam. Unsubscribe anytime. 🔒 Privacy first.
          </p>
        </motion.div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 9. FOOTER (Chordify style)                                         */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <footer
        id="contact"
        style={{
          background: '#0c0c1a',
          borderTop: '1px solid rgba(255,255,255,0.07)',
        }}
      >
        {/* ── Main footer body ── */}
        <div style={{
          maxWidth: 1200, margin: '0 auto',
          padding: '64px 32px 48px',
          display: 'grid',
          gridTemplateColumns: '240px 1fr',
          gap: 64,
          alignItems: 'start',
        }}>

          {/* LEFT: Logo + tagline + social + app badges */}
          <div>
            {/* Logo + name */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
              <LogoIcon size={36} />
              <span style={{
                fontFamily: "'Outfit', sans-serif",
                fontSize: 20, fontWeight: 800, color: '#fff',
                letterSpacing: '-0.02em',
              }}>
                ChordSense<span style={{
                  background: 'linear-gradient(90deg,#a78bfa,#67e8f9)',
                  WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                }}>Pro</span>
              </span>
            </div>

            <p style={{
              color: 'rgba(255,255,255,0.4)', fontSize: 13, lineHeight: 1.6,
              margin: '0 0 20px', maxWidth: 200,
            }}>
              Phân tích sâu — Luyện tập chuẩn — Thành thạo toàn diện
            </p>

            {/* Social icons */}
            <div style={{ display: 'flex', gap: 10, marginBottom: 24 }}>
              {[
                { title: 'Instagram', svg: <svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="2" width="20" height="20" rx="5"/><path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"/><line x1="17.5" y1="6.5" x2="17.51" y2="6.5"/></svg> },
                { title: 'TikTok', svg: <svg width={18} height={18} viewBox="0 0 24 24" fill="currentColor"><path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-2.88 2.5 2.89 2.89 0 0 1-2.89-2.89 2.89 2.89 0 0 1 2.89-2.89c.28 0 .54.04.79.1V9.01a6.33 6.33 0 0 0-.79-.05 6.34 6.34 0 0 0-6.34 6.34 6.34 6.34 0 0 0 6.34 6.34 6.34 6.34 0 0 0 6.33-6.34V8.69a8.26 8.26 0 0 0 4.84 1.56V6.81a4.85 4.85 0 0 1-1.07-.12z"/></svg> },
                { title: 'Facebook', svg: <svg width={18} height={18} viewBox="0 0 24 24" fill="currentColor"><path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z"/></svg> },
                { title: 'Discord', svg: <svg width={18} height={18} viewBox="0 0 24 24" fill="currentColor"><path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03z"/></svg> },
              ].map(({ title, svg }) => (
                <motion.a
                  key={title}
                  href="#"
                  title={title}
                  whileHover={{ scale: 1.12, color: '#a78bfa' }}
                  style={{
                    width: 36, height: 36, borderRadius: 8,
                    border: '1px solid rgba(255,255,255,0.12)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: 'rgba(255,255,255,0.45)', textDecoration: 'none',
                    transition: 'border-color 0.2s, color 0.2s',
                    background: 'rgba(255,255,255,0.04)',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(167,139,250,0.5)'; e.currentTarget.style.color = '#a78bfa' }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)'; e.currentTarget.style.color = 'rgba(255,255,255,0.45)' }}
                >
                  {svg}
                </motion.a>
              ))}

          </div>
          </div>

          {/* RIGHT: 4 nav columns */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: 32,
          }}>
            {[
              {
                title: 'Discover',
                links: [
                  { label: 'AI Chord Engine' },
                  { label: 'Smart Grid View' },
                  { label: 'Featured Songs' },
                  { label: 'Music Dictionary' },
                  { label: 'Chord Diagram Finder' },
                  { label: 'Blog' },
                ],
              },
              {
                title: 'Product',
                links: [
                  { label: 'Pro' },
                  { label: 'Harmonic Analysis' },
                  { label: 'A/B Loop Tool' },
                  { label: 'Piano Mode' },
                  { label: 'Guitar Mode' },
                ],
              },
              {
                title: 'Support',
                links: [
                  { label: 'Help Center' },
                  { label: 'Community', highlight: true },
                  { label: 'Contact Us' },
                ],
              },
              {
                title: 'About',
                links: [
                  { label: 'About ChordSense' },
                  { label: 'HCMUT Research' },
                  { label: 'Press' },
                  { label: 'Terms & Conditions' },
                  { label: 'Privacy Statement' },
                  { label: 'Your Privacy Settings' },
                ],
              },
            ].map(({ title, links }) => (
              <div key={title}>
                <h4 style={{
                  color: '#fff', fontSize: 13, fontWeight: 700,
                  margin: '0 0 16px', letterSpacing: '0.02em',
                }}>
                  {title}
                </h4>
                <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'flex', flexDirection: 'column', gap: 9 }}>
                  {links.map(({ label, highlight }) => (
                    <li key={label}>
                      <a
                        href="#"
                        style={{
                          color: highlight ? '#a78bfa' : 'rgba(255,255,255,0.4)',
                          fontSize: 13, textDecoration: 'none',
                          transition: 'color 0.2s',
                          fontWeight: 400,
                        }}
                        onMouseEnter={e => e.target.style.color = highlight ? '#c4b5fd' : 'rgba(255,255,255,0.8)'}
                        onMouseLeave={e => e.target.style.color = highlight ? '#a78bfa' : 'rgba(255,255,255,0.4)'}
                      >
                        {label}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        {/* ── Bottom bar ── */}
        <div style={{
          borderTop: '1px solid rgba(255,255,255,0.06)',
          padding: '18px 32px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          flexWrap: 'wrap', gap: 16,
          maxWidth: 1200, margin: '0 auto',
        }}>
          {/* Payment icons */}
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            {[
              { name: 'Visa', bg: '#1a1f71', color: '#fff', text: 'VISA', w: 44 },
              { name: 'Mastercard', bg: '#eb001b', color: '#fff', text: 'MC', w: 36, extra: '#f79e1b' },
              { name: 'PayPal', bg: '#003087', color: '#009cde', text: 'Pay', w: 44 },
              { name: 'Momo', bg: '#ae2070', color: '#fff', text: 'MoMo', w: 48 },
              { name: 'VNPay', bg: '#1a56db', color: '#fff', text: 'VNPay', w: 52 },
            ].map(({ name, bg, color, text, w }) => (
              <div key={name} style={{
                width: w, height: 24, borderRadius: 4,
                background: bg,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 9, fontWeight: 800, color, letterSpacing: '0.03em',
                flexShrink: 0,
              }}>
                {text}
              </div>
            ))}
          </div>

          {/* Made with love */}
          <p style={{
            color: 'rgba(255,255,255,0.3)', fontSize: 12, margin: 0,
            display: 'flex', alignItems: 'center', gap: 5,
          }}>
            Made with{' '}
            <span style={{ color: '#f472b6', fontSize: 14 }}>♥</span>
            {' '}in Ho Chi Minh City, Vietnam
          </p>
        </div>
      </footer>

    </div>
  )
}
