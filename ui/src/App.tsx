import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { Header, HeaderName, SideNav, SideNavItems, SideNavLink } from '@carbon/react'
import ApiStatus from './components/ApiStatus'
import PageHeader from './components/PageHeader'
import { SCREENS } from './screens/registry'

function Rail() {
  const { pathname } = useLocation()
  const navigate = useNavigate()
  return (
    <SideNav isFixedNav expanded isChildOfHeader={false} aria-label="Screens">
      <SideNavItems>
        {SCREENS.map((s) => (
          <SideNavLink
            key={s.id}
            href={s.path}
            renderIcon={s.icon}
            isActive={pathname === s.path}
            onClick={(e: React.MouseEvent) => {
              e.preventDefault()
              navigate(s.path)
            }}
          >
            {s.title}
          </SideNavLink>
        ))}
      </SideNavItems>
    </SideNav>
  )
}

export default function App() {
  const navigate = useNavigate()
  return (
    <>
      <Header aria-label="Demand forecasting">
        <HeaderName
          href="/"
          prefix="AnvIQ"
          onClick={(e: React.MouseEvent) => {
            e.preventDefault()
            navigate('/overview')
          }}
        >
          Demand forecasting
        </HeaderName>
      </Header>
      <Rail />
      <main className="app-content">
        <Routes>
          <Route path="/" element={<Navigate to="/overview" replace />} />
          {SCREENS.map((s) => (
            <Route
              key={s.id}
              path={s.path}
              element={
                <>
                  <PageHeader group={s.group} title={s.title} subtitle={s.subtitle} right={<ApiStatus />} />
                  <s.Component />
                </>
              }
            />
          ))}
          <Route path="*" element={<Navigate to="/overview" replace />} />
        </Routes>
      </main>
    </>
  )
}
