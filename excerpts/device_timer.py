start_event, end_event, event_scale = self.events[buffer].get(name, event_info)
start_event.synchronize()
end_event.synchronize()
t = start_event.elapsed_time(end_event)
s = event_scale if event_scale is not None else 1.0
t = t / 1000 * s
