from stunes.ingest import consumer as c


class FakeConsumer:
    def __init__(self, parts):
        self.parts = parts

    def assignment(self):
        return self.parts


def _age(clock, s):
    clock.last -= s
    clock.assigned_at -= s


def test_idle_counts_only_while_partitions_are_held():
    clock = c.IdleClock(5)
    unassigned, assigned = FakeConsumer([]), FakeConsumer(["p0"])
    _age(clock, 30)  # 30 s waiting for the group to hand out partitions
    assert not clock.expired(unassigned)  # the old clock would have quit here with 0 messages
    assert not clock.expired(assigned)  # just got a partition: idle starts now
    _age(clock, 6)
    assert clock.expired(assigned)


def test_never_assigned_gives_up_after_the_startup_limit():
    clock = c.IdleClock(5, unassigned_s=60)
    _age(clock, 61)
    assert clock.expired(FakeConsumer([]))


def test_retry_consumer_waits_for_the_main_consumers(tmp_path):
    flag = tmp_path / "main-done"
    clock = c.IdleClock(8, after=flag)
    assigned = FakeConsumer(["r0"])
    _age(clock, 100)  # a slow run: poison arrives far apart
    assert not clock.expired(assigned)
    flag.touch()
    assert not clock.expired(assigned)  # idling counts from the flag on
    _age(clock, 9)
    assert clock.expired(assigned)


def test_no_idle_limit_never_expires():
    clock = c.IdleClock(None)
    _age(clock, 10_000)
    assert not clock.expired(FakeConsumer([]))


def test_unassigned_limit_also_waits_for_the_main_consumers(tmp_path):
    flag = tmp_path / "main-done"
    clock = c.IdleClock(8, after=flag, unassigned_s=60)
    unassigned = FakeConsumer([])
    _age(clock, 600)  # a long run
    assert not clock.expired(unassigned)
    flag.touch()
    assert not clock.expired(unassigned)  # not expired the moment the flag appears
    _age(clock, 61)
    assert clock.expired(unassigned)
