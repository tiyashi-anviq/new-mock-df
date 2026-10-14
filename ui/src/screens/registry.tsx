import type { ComponentType } from 'react'
import { Book, ChartLine, Dashboard } from '@carbon/icons-react'
import Overview from './Overview'
import Forecast from './Forecast'
import Guide from './Guide'

export interface ScreenDef {
  id: string
  path: string
  title: string
  subtitle: string
  group: 'Plan' | 'Assist'
  icon: ComponentType
  Component: ComponentType
}

// One list drives the routes, the left rail and the page header. To add a screen, add a row here.
export const SCREENS: ScreenDef[] = [
  { id: 'overview', path: '/overview', title: 'Overview', subtitle: 'Where the data stands and what is ready to show.', group: 'Plan', icon: Dashboard, Component: Overview },
  { id: 'forecast', path: '/forecast', title: 'Forecast & accuracy', subtitle: 'The forecast, and how accurate it was one, two and three months ahead.', group: 'Plan', icon: ChartLine, Component: Forecast },
  { id: 'guide', path: '/guide', title: 'Guide', subtitle: 'How to read the screens.', group: 'Assist', icon: Book, Component: Guide },
]
