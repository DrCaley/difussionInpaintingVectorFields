import unittest
import ddpm.vector_combination.combination_loss


class MyTestCase(unittest.TestCase):
    def test_combination_loss_module_exists(self):
        """Verify the combination_loss module can be imported."""
        self.assertTrue(hasattr(ddpm.vector_combination.combination_loss, '__name__'))


if __name__ == '__main__':
    unittest.main()
