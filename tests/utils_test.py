import unittest
from unittest import mock

from slippi_ai import utils


class UtilsPortRangeTest(unittest.TestCase):

  def test_get_slippi_port_range_defaults_to_bounded_high_range(self):
    with mock.patch.object(utils, 'SLIPPI_PORT_MIN', 25000), \
         mock.patch.object(utils, 'SLIPPI_PORT_MAX', 65535):
      self.assertEqual(utils.get_slippi_port_range(), (25000, 65535))

  def test_get_slippi_port_range_rejects_invalid_bounds(self):
    with mock.patch.object(utils, 'SLIPPI_PORT_MIN', 65000), \
         mock.patch.object(utils, 'SLIPPI_PORT_MAX', 64000):
      with self.assertRaisesRegex(ValueError, 'Invalid port range'):
        utils.get_slippi_port_range()

  def test_find_open_udp_ports_stays_within_configured_range(self):
    fake_netstat = (
        "Active Internet connections\n"
        "Proto Recv-Q Send-Q Local Address           Foreign Address         State\n"
        "udp        0      0 0.0.0.0:25001          0.0.0.0:*\n"
    ).encode()
    with mock.patch.object(utils, 'SLIPPI_PORT_MIN', 25000), \
         mock.patch.object(utils, 'SLIPPI_PORT_MAX', 25003), \
         mock.patch.object(utils.platform, 'system', return_value='Linux'), \
         mock.patch.object(utils.subprocess, 'check_output', return_value=fake_netstat):
      ports = utils.find_open_udp_ports(3)
    self.assertEqual(set(ports), {25000, 25002, 25003})


if __name__ == '__main__':
  unittest.main()
